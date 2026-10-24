# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Lock-step SGLang/FlashInfer 0.7.1 norm worker.

Parent CLI: norm_act_paired --market-root DIR [--fi071 INSTALL_DIR]. Without
--fi071, install a private --no-deps overlay in a temp dir. Internal worker CLI:
python -m benchmark.kernel_tuning.norm_act_market --worker --market-root DIR.
Only protocol replies go to stdout; imports, builds and version reports use stderr.
"""

import argparse
import base64
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import types
from dataclasses import asdict
from pathlib import Path

import torch
from torch.multiprocessing import reductions

from benchmark.kernel_tuning.norm_act_guard import install_profiler_guard, profiler_clean
from benchmark.kernel_tuning.norm_act_pools import (
    POOL_BASES,
    norm_pool_views,
)
from benchmark.kernel_tuning.norm_act_probe import run_probe
from benchmark.kernel_tuning.norm_act_vllm import MarketError

PROTOCOL_VERSION = 2
COMMANDS = {"setup", "check", "warm", "sample", "profile", "free", "exit"}


class CudaIPCError(MarketError):
    def __init__(self):
        super().__init__("CUDA IPC pool sharing failed; private worker pools are not supported")


def encode_pool_handles(pools):
    """Export every allocation, including restore sources; caller retains pools."""
    try:
        handles = {}
        for name in POOL_BASES:
            tensor = getattr(pools, name)
            if tensor is None:
                handles[name] = None
                continue
            if tensor.device.type != "cuda":
                raise CudaIPCError()
            rebuild, args = reductions.reduce_tensor(tensor)
            handles[name] = base64.b64encode(
                pickle.dumps((rebuild.__module__, rebuild.__name__, args))
            ).decode("ascii")
        return handles
    except Exception:  # noqa: BLE001 -- redact native errors and never fall back
        raise CudaIPCError() from None


def decode_pool_handles(spec, handles):
    """Decode only the trusted parent pipe's pickle payload, never external input."""
    try:
        if set(handles) != set(POOL_BASES):
            raise CudaIPCError()
        bases = {}
        for name in POOL_BASES:
            encoded = handles[name]
            if encoded is None:
                if name != "aux" or spec["op"].endswith("_sum"):
                    raise CudaIPCError()
                bases[name] = None
                continue
            module, function, args = pickle.loads(base64.b64decode(encoded, validate=True))
            rebuild = getattr(importlib.import_module(module), function)
            bases[name] = rebuild(*args)
            if bases[name].device.type != "cuda":
                raise CudaIPCError()
        device = bases["x"].device
        if any(t is not None and t.device != device for t in bases.values()):
            raise CudaIPCError()
        torch.cuda.set_device(device)
        return norm_pool_views(spec, **bases)
    except Exception:  # noqa: BLE001 -- redact native errors and never fall back
        raise CudaIPCError() from None


def encode_message(message):
    # Encode nonfinite error metrics explicitly, not as non-standard JSON literals.
    def encode(value):
        if isinstance(value, float) and not math.isfinite(value):
            return {"$float": repr(value)}
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        return value

    return json.dumps(encode(message), separators=(",", ":"), allow_nan=False) + "\n"


def decode_message(line):
    def decode(value):
        if set(value) == {"$float"}:
            if value["$float"] not in ("inf", "-inf", "nan"):
                raise ValueError("invalid nonfinite metric")
            return float(value["$float"])
        return value

    result = json.loads(line, object_hook=decode)
    if not isinstance(result, dict):
        raise ValueError("protocol message must be an object")
    return result


def sample_us(elapsed_ms, layers):
    if not elapsed_ms or layers <= 0 or any(not math.isfinite(t) or t < 0 for t in elapsed_ms):
        raise ValueError("invalid graph timing")
    return sum(elapsed_ms) * 1000 / (layers * len(elapsed_ms))


def load_sglang_norm(root):
    import cutlass  # noqa: F401 -- verify CuTe DSL imports in the isolated process
    import flashinfer
    import flashinfer.norm
    from packaging.version import Version

    from benchmark.kernel_tuning.norm_act_norm import flashinfer_metadata
    from benchmark.kernel_tuning.norm_act_vllm import load_file

    version = importlib.metadata.version("flashinfer-python")
    if version != "0.7.1" or getattr(flashinfer, "__version__", version) != "0.7.1":
        raise RuntimeError("worker requires FlashInfer 0.7.1")
    cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
    if Version(cutlass_version) < Version("4.7"):
        raise RuntimeError("FlashInfer 0.7.1 norm requires cutlass-dsl >=4.7")
    package = types.ModuleType("sgl_kernel")
    package.__path__ = [str(root / "sgl_kernel")]
    sys.modules["sgl_kernel"] = package
    hashes = {}
    for name in ("utils", "elementwise"):
        path = root / "sgl_kernel" / f"{name}.py"
        module = load_file(f"sgl_kernel.{name}", path)
        setattr(package, name, module)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if module._has_flashinfer is not True:
        raise RuntimeError("SGLang wrapper did not select the FlashInfer branch")
    metadata = flashinfer_metadata() | {
        "cutlass_dsl": cutlass_version,
        "torch": torch.__version__,
        "wrapper_sha256": hashes,
        "sglang_revision": "72730d6",
        "sglang_kernel_version": "0.4.9",
        # SGLang 72730d6 python/sglang/srt/layers/layernorm.py RMSNorm.forward_cuda
        # and GemmaRMSNorm._forward_impl select these functions for these inputs.
        "dispatch_chain": "SGLang 72730d6 python/sglang/srt/layers/layernorm.py:621 / 1132 -> sglang-kernel 0.4.9 elementwise.py:125-162 / 206-245 -> flashinfer.norm.fused_add_rmsnorm / gemma_fused_add_rmsnorm (0.7.1, enable_pdl=None)",
        "upstream_pins": {"flashinfer_python[cu13]": "0.7.1", "torch": "2.14.1"},
    }
    print(
        f"market worker: FlashInfer {version}, cutlass-dsl {cutlass_version}, torch {torch.__version__}",
        file=sys.stderr,
        flush=True,
    )
    return module, flashinfer.norm, metadata


class Worker:
    def __init__(self, root):
        install_profiler_guard()
        self.root = root
        self.wrapper, self.direct, self.metadata = load_sglang_norm(root)
        self.pools = None
        self.banks = {}

    def dispatch(self, command, payload):
        from benchmark.kernel_tuning.norm_act_norm import (
            NormCall,
            check_norm_candidates,
            check_sum_norm_candidates,
        )
        from benchmark.kernel_tuning.norm_act_paired import (
            SumNormCall,
            capture_bank,
        )
        from ctikernels.norm._dispatch import Config

        if command == "setup":
            if self.pools is not None:
                raise ValueError("free previous row before setup")
            self.spec = payload["spec"]
            self.cute_config = payload["cute_config"]
            self.layers = self.spec["layers"]
            self.pools = decode_pool_handles(self.spec, payload.get("pool_handles", {}))
            # The protocol is lock-step: the parent is idle throughout each worker
            # command and vice versa. Never mutate IPC pools outside dispatch/sample.
            zero = self.spec["op"].startswith("gemma")
            if self.spec["op"].endswith("_sum"):
                self.metadata["aux_capture"] = (
                    "SGLang 72730d6 layer_boundary/residual/access.py:final_norm_pair / "
                    "layer_boundary/stage.py -> capture(residual); "
                    "layers/aux_hidden_states.py:AuxHiddenStateList.capture clones; "
                    "preallocated packer copies into [tokens, K*hidden]; "
                    "benchmark: even norm, odd norm then aux_slice.copy_(residual)"
                )
            name = "gemma_fused_add_rmsnorm" if zero else "fused_add_rmsnorm"
            cute = NormCall(
                zero,
                "cute",
                Config(**payload["cute_config"]),
                static_weight=self.spec["static_weight"],
            )
            self.candidates = {
                "cute_remote": SumNormCall(cute, fused=True)
                if self.spec["op"].endswith("_sum")
                else cute
            }
            for candidate, module in (("sglang", self.wrapper), ("fi071_direct", self.direct)):
                self.candidates[candidate] = SglangNormCall(getattr(module, name))
            for call in self.candidates.values():
                for index in (0, 1):
                    self.pools.restore()
                    if hasattr(call, "call_index"):
                        call.call_index(self.pools.tensors[0], None, index)
                    else:
                        call(self.pools.tensors[0])
            self.pools.restore()
            torch.cuda.synchronize()
            return {
                "metadata": self.metadata,
                "count": len(self.pools.inputs),
                "distinct_weights": len(self.pools.tensors),
                "shared_buffers": True,
            }
        if command in ("free", "exit"):
            torch.cuda.synchronize()
            self.banks.clear()
            self.pools = None
            self.candidates = {}
            gc.collect()
            # Release graphs, retained outputs and imported storage before the ack.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return {}
        if self.pools is None:
            raise ValueError("setup required")
        if command == "check":
            checker = (
                check_sum_norm_candidates
                if self.spec["op"].endswith("_sum")
                else check_norm_candidates
            )
            return checker(
                self.candidates,
                self.pools.tensors[
                    payload.get("start", 0) : payload.get("stop", len(self.pools.tensors))
                ],
                self.spec["op"].startswith("gemma"),
                record_inputs=True,
                shared_pools=self.pools,
            )
        if command == "profile":
            return run_probe(
                {
                    "kind": "worker",
                    "candidate": payload["candidate"],
                    "market_root": self.root,
                    "config": self.cute_config,
                    **{
                        key: self.spec[key]
                        for key in ("op", "m", "width", "dtype", "static_weight")
                    },
                }
            )
        if command == "warm":
            if not self.banks:
                for name, fn in self.candidates.items():
                    self.pools.restore()
                    self.banks[name] = capture_bank(
                        fn,
                        self.pools.inputs,
                        [None] * len(self.pools.inputs),
                        self.layers,
                        warm_restore=self.pools.restore,
                    )
            for name in self.banks:
                self.sample(name)
            return {}
        if command == "sample":
            return {"us": self.sample(payload["candidate"]), "profiler_clean": profiler_clean()}
        raise ValueError("unknown worker command")

    def sample(self, name):
        profiler_clean()
        self.pools.restore()
        graphs = self.banks[name][0]
        for graph, _, _ in graphs:
            graph.replay()
        torch.cuda.synchronize()
        return sample_us([begin.elapsed_time(end) for _, begin, end in graphs], self.layers)


class SglangNormCall:
    """SGLang captures the updated residual after norm, on odd layers only."""

    def __init__(self, operation):
        self.operation = operation

    def __call__(self, tensors, unused=None):
        x, residual, weight = tensors[:3]
        self.operation(x, residual, weight, 1e-6, enable_pdl=None)
        if len(tensors) == 4:
            # SGLang 72730d6: layer_boundary/residual/access.py:final_norm_pair
            # and layer_boundary/stage.py call capture(residual). In
            # layers/aux_hidden_states.py AuxHiddenStateList.capture clones it;
            # the preallocated packer copies it into [tokens, K * hidden].
            tensors[3].copy_(residual)
        return x, residual

    def call_index(self, tensors, unused, index):
        return self(tensors[:3] if index % 2 == 0 else tensors)


class MarketWorker:
    def __init__(self, root, target):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmark.kernel_tuning.norm_act_market",
                "--worker",
                "--market-root",
                str(root.resolve()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.sequence = 0
        self.closed = False
        self.shared_pools = None

    def request(self, command, **payload):
        if command not in COMMANDS:
            raise ValueError("unknown worker command")
        torch.cuda.synchronize()
        self.sequence += 1
        self.process.stdin.write(
            encode_message(
                {"version": PROTOCOL_VERSION, "id": self.sequence, "command": command, **payload}
            )
        )
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise MarketError("market worker closed its protocol stream")
        reply = decode_message(line)
        if reply.get("id") != self.sequence or reply.get("version") != PROTOCOL_VERSION:
            raise MarketError("market worker protocol sequence/version mismatch")
        if not reply.get("ok"):
            if reply.get("error_type") == "CudaIPCError":
                raise CudaIPCError()
            raise MarketError(
                "market worker command failed: "
                + reply.get("error_type", "UnknownError")
                + (f" ({reply['error']})" if reply.get("error") else "")
            )
        if command in ("free", "exit"):
            # The worker has dropped every IPC reference and synchronized before ack.
            self.shared_pools = None
        return reply["result"]

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.poll() is None:
                if self.shared_pools is not None:
                    self.request("free")
                self.request("exit")
        finally:
            self.process.stdin.close()
            self.process.stdout.close()
            # Cooperative EOF/exit only. Never terminate or kill a process.
            self.process.wait()
            # EOF/error cleanup also keeps the producer alive until consumer exit.
            self.shared_pools = None


def start_market(args, pools, *, cute_config):
    """Reuse the private installation, never an interpreter containing FI 0.6."""
    from benchmark.kernel_tuning.norm_act_fi07 import install_flashinfer

    # Fail before spawning a consumer if this device cannot export CUDA IPC.
    handles = encode_pool_handles(pools)
    target = getattr(args, "fi071", None)
    if target is None:
        target = Path(tempfile.mkdtemp(prefix="cti-fi071-", dir="/var/tmp"))
        with contextlib.redirect_stdout(sys.stderr):
            install_flashinfer(target, "0.7.1")
        args.fi071 = target
    worker = MarketWorker(args.market_root, target)
    worker.shared_pools = pools
    try:
        setup = worker.request(
            "setup",
            spec=pools.spec,
            cute_config=asdict(cute_config),
            pool_handles=handles,
        )
        if setup.get("shared_buffers") is not True:
            raise CudaIPCError()
    except BaseException:
        worker.close()
        raise
    return worker, setup


class RemoteCandidate:
    def __init__(self, worker, name):
        self.worker, self.name = worker, name

    def warm(self):
        self.worker.request("warm")

    def sample(self):
        value = self.worker.request("sample", candidate=self.name)["us"]
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid remote timing")
        return value


def main(argv=None):
    install_profiler_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    args = parser.parse_args(argv)
    # Native build output also goes to stderr. Keep a separate protocol descriptor.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
        worker = None
        for line in sys.stdin:
            request = {}
            try:
                request = decode_message(line)
                if (
                    request.get("version") != PROTOCOL_VERSION
                    or request.get("command") not in COMMANDS
                ):
                    raise ValueError("invalid protocol request")
                if worker is None:
                    worker = Worker(args.market_root)
                result = worker.dispatch(request["command"], request)
                torch.cuda.synchronize()
                reply = {"ok": True, "result": result}
            except Exception as error:  # noqa: BLE001 -- keep protocol alive, redact native errors
                # Exception messages from pip/imports can contain environment values;
                # the protocol intentionally reports only the exception type.
                reply = {"ok": False, "error_type": type(error).__name__}
                if isinstance(error, MarketError):
                    # Our own messages carry no environment values; keep them for receipts.
                    reply["error"] = str(error)[:300]
            protocol.write(
                encode_message({"version": PROTOCOL_VERSION, "id": request.get("id"), **reply})
            )
            if request.get("command") == "exit":
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
