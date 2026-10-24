# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Exchange: rank-ordered sums bitwise against the FP32 reference and across ranks, the norm
exit, the GEMM-epilogue publish hook, graph replay, zero rows and abort. Ranks share GPUs
round-robin, so two ranks run on one device too."""

import queue as std_queue
import socket
import time
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl

from ctikernels.collective.inference import Exchange, reference
from ctikernels.collective.triton.exchange import publish
from ctikernels.linear.lora import lora_shrink
from ctikernels.linear.triton._block_fp8 import gemm
from ctikernels.moe_impl import moe_sum_reduce_triton

WORLD = 2


@triton.jit
def _tile_producer(xch, x, out, M, N, TM: tl.constexpr, TN: tl.constexpr, RANK: tl.constexpr,
                   WORLD: tl.constexpr):  # fmt: skip
    """A GEMM-shaped producer: out = 2 * x per TM x TN tile, published from the epilogue."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rows = pid_m * TM + tl.arange(0, TM)
    cols = pid_n * TN + tl.arange(0, TN)
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    offs = rows[:, None] * N + cols[None, :]
    values = (tl.load(x + offs, mask=mask, other=0.0) * 2).to(out.dtype.element_ty)
    tl.store(out + offs, values, mask=mask)
    publish(xch, values, pid_m * TM, pid_n * TN, M, N, pid_m * tl.num_programs(1) + pid_n, RANK,
            WORLD)  # fmt: skip


def _parts(rank, tokens, hidden, seed):
    """Every rank's partial (CPU, bfloat16), the same on every rank."""
    return [
        torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(seed + r)).bfloat16()
        for r in range(WORLD)
    ]


def _rank_sum(parts, residual=None):
    h = torch.zeros(parts[0].shape) if residual is None else residual.float().clone()
    for p in parts:
        h = h + p.float()
    return h


def _same_on_all_ranks(tensor):
    gathered = [torch.empty_like(tensor) for _ in range(WORLD)]
    dist.all_gather(gathered, tensor.contiguous())
    return all(torch.equal(g, gathered[0]) for g in gathered)


def _worker(rank, port, transport, queue):
    errors = []
    try:
        device = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
        dist.init_process_group(
            "gloo", rank=rank, world_size=WORLD, init_method=f"tcp://127.0.0.1:{port}",
            timeout=timedelta(minutes=10),
        )  # fmt: skip
        xch = Exchange(
            rank=rank, world_size=WORLD, cpu_group=dist.group.WORLD, device=device,
            max_tokens=512, max_hidden=5120, transport=transport,
        )  # fmt: skip
        if xch.transport != transport:
            errors.append(f"transport {xch.transport} != {transport}")

        def expect(label, ok):
            if not ok:
                errors.append(label)

        for tokens, hidden in ((1, 2048), (3, 5120), (4, 4096), (64, 2048), (1100, 5120)):
            parts = _parts(rank, tokens, hidden, 31 * tokens + hidden)
            x = parts[rank].to(device)
            xch.all_reduce(x)
            want = _rank_sum(parts).bfloat16()
            expect(f"all_reduce T={tokens} H={hidden}", torch.equal(x.cpu(), want))

        for zero_centered in (True, False):
            for tokens, hidden in ((1, 4096), (4, 5120), (600, 2048)):
                parts = _parts(rank, tokens, hidden, 7 * tokens + hidden)
                g = torch.Generator().manual_seed(hidden + tokens)
                residual = torch.randn(tokens, hidden, generator=g).bfloat16()
                weight = torch.randn(hidden, generator=g).bfloat16()
                x, r = parts[rank].to(device), residual.to(device)
                xch.add_rms_norm(x, r, weight.to(device), 1e-6, zero_centered)
                ref_x, ref_r = reference(
                    torch.stack(parts), residual, weight, 1e-6, zero_centered=zero_centered
                )
                label = f"add_rms_norm zc={zero_centered} T={tokens} H={hidden}"
                expect(f"{label} residual", torch.equal(r.cpu(), ref_r))
                close = torch.allclose(x.cpu().float(), ref_x.float(), rtol=1e-2, atol=1e-2)
                expect(f"{label} x", close)
                expect(f"{label} ranks", _same_on_all_ranks(x.cpu()))

        # The epilogue hook: 2-D tiles from a producer kernel, then the fused reduce + norm.
        for tokens, hidden, tm, tn in (
            (1, 5120, 16, 128),
            (40, 2048, 16, 64),
            (512, 4096, 64, 128),
        ):
            parts = _parts(rank, tokens, hidden, tokens + 3 * hidden)
            g = torch.Generator().manual_seed(tokens)
            residual = torch.randn(tokens, hidden, generator=g).bfloat16()
            weight = torch.randn(hidden, generator=g).bfloat16()
            out = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device)
            r = residual.to(device)
            ep = xch.epilogue
            grid = (triton.cdiv(tokens, tm), triton.cdiv(hidden, tn))
            _tile_producer[grid](ep.desc, parts[rank].to(device), out, tokens, hidden, TM=tm,
                                 TN=tn, RANK=ep.rank, WORLD=ep.world)  # fmt: skip
            xch.reduce(out, tile=(tm, tn), residual=r, norm=(weight.to(device), 1e-6, True))
            doubled = [p.float().mul(2).bfloat16() for p in parts]
            ref_x, ref_r = reference(torch.stack(doubled), residual, weight, 1e-6)
            label = f"hook T={tokens} H={hidden} tile={tm}x{tn}"
            expect(f"{label} residual", torch.equal(r.cpu(), ref_r))
            close = torch.allclose(out.cpu().float(), ref_x.float(), rtol=1e-2, atol=1e-2)
            expect(f"{label} x", close)
            expect(f"{label} ranks", _same_on_all_ranks(out.cpu()))

        # Row-parallel BF16 GEMM + LoRA (K sharded): the fused epilogue equals GEMM-then-exchange.
        for tokens in (1, 4, 300):
            for config in ((16, 64, 1, 4, 3), (16, 64, 2, 4, 3)):
                g = torch.Generator().manual_seed(tokens)
                k_shard, n, rank_r = 1024, 2048, 16
                cols = slice(rank * k_shard, (rank + 1) * k_shard)

                def randn(*shape, scale=1.0, g=g):
                    return (torch.randn(*shape, generator=g) * scale).bfloat16().to(device)

                x = randn(tokens, WORLD * k_shard)[:, cols].contiguous()
                w = randn(n, WORLD * k_shard, scale=0.03)[:, cols].contiguous()
                factors = (randn(rank_r, WORLD * k_shard, scale=0.03)[:, cols].contiguous(),
                           randn(n, rank_r, scale=0.03))  # fmt: skip
                residual, gain = randn(tokens, n), randn(n)
                results = []
                for fused in (False, True):
                    out, r = (
                        torch.empty(tokens, n, dtype=torch.bfloat16, device=device),
                        residual.clone(),
                    )
                    lora = lora_shrink(
                        x, factors, scaling=0.5, output_sizes=(n,), row_parallel=True
                    )
                    ep = xch.epilogue if fused else None
                    tile = gemm(x, None, w, None, out, config=config, lora=lora, exchange=ep)
                    if fused:
                        xch.reduce(out, tile=tile, residual=r, norm=(gain, 1e-6, True))
                    else:
                        xch.add_rms_norm(out, r, gain, 1e-6)
                    results.append((out.cpu(), r.cpu()))
                label = f"gemm+lora T={tokens} config={config}"
                expect(f"{label} x", torch.equal(results[0][0], results[1][0]))
                expect(f"{label} residual", torch.equal(results[0][1], results[1][1]))
                expect(f"{label} ranks", _same_on_all_ranks(results[1][0]))

        # MoE exit (EP): the slot sum publishes; fused equals slot-sum-then-exchange.
        for tokens in (1, 3, 40):
            g = torch.Generator().manual_seed(5 * tokens + rank)
            routes = torch.randn(tokens, 4, 2048, generator=g).to(device)
            residual = torch.randn(tokens, 2048, generator=g).bfloat16().to(device)
            gain = torch.randn(2048, generator=g).bfloat16().to(device)
            results = []
            for fused in (False, True):
                out, r = torch.empty_like(residual), residual.clone()
                tile = moe_sum_reduce_triton(routes, out, exchange=xch.epilogue if fused else None)
                if fused:
                    xch.reduce(out, tile=tile, residual=r, norm=(gain, 1e-6, False))
                else:
                    xch.add_rms_norm(out, r, gain, 1e-6, False)
                results.append((out.cpu(), r.cpu()))
            label = f"moe exit T={tokens}"
            expect(f"{label} x", torch.equal(results[0][0], results[1][0]))
            expect(f"{label} residual", torch.equal(results[0][1], results[1][1]))

        # A small exchange: word-protocol slots sized for the capacity.
        small = Exchange(
            rank=rank, world_size=WORLD, cpu_group=dist.group.WORLD, device=device,
            max_tokens=4, max_hidden=4096, transport=transport,
        )  # fmt: skip
        parts = _parts(rank, 4, 4096, 11)
        x = parts[rank].to(device)
        small.all_reduce(x)
        expect("small exchange", torch.equal(x.cpu(), _rank_sum(parts).bfloat16()))
        dist.barrier()
        small.close()

        # Calls beyond the capacity raise on every rank before anything is sent (a publishing
        # producer skips them), and the next call still pairs up.
        wide = torch.zeros(2, 8192, dtype=torch.bfloat16, device=device)
        try:
            xch.all_reduce(wide)
        except ValueError:
            pass
        else:
            errors.append("all_reduce beyond max_hidden did not raise")
        ep = xch.epilogue
        _tile_producer[(1, 64)](ep.desc, wide, wide, 2, 8192, TM=16, TN=128, RANK=ep.rank,
                                WORLD=ep.world)  # fmt: skip
        try:
            xch.reduce(wide, tile=(16, 128))
        except ValueError:
            pass
        else:
            errors.append("reduce beyond max_hidden did not raise")
        # A producer that published, then a reduce rejected on layout (odd row stride): the
        # next call of the same shape must not read the rejected call's words.
        stale = torch.full((4, 4100), 7.0, dtype=torch.bfloat16, device=device)[:, :4096]
        _tile_producer[(1, 32)](ep.desc, stale.contiguous(), stale.contiguous(), 4, 4096, TM=16,
                                TN=128, RANK=ep.rank, WORLD=ep.world)  # fmt: skip
        try:
            xch.reduce(stale, tile=(16, 128))
        except ValueError:
            pass
        else:
            errors.append("reduce with an odd row stride did not raise")
        parts = _parts(rank, 4, 4096, 77)
        x = parts[rank].to(device)
        xch.all_reduce(x)
        expect("in step after rejected calls", torch.equal(x.cpu(), _rank_sum(parts).bfloat16()))

        # Zero rows launch nothing; the next call still pairs up.
        xch.all_reduce(torch.empty(0, 2048, dtype=torch.bfloat16, device=device))

        # Graph replay mixed with eager calls: all-reduce (word protocol), a sliced call (flags),
        # add + RMSNorm, and a producer followed by the fused reduce.
        parts = _parts(rank, 4, 4096, 99)
        big = _parts(rank, 1100, 2048, 98)
        g = torch.Generator().manual_seed(97)
        residual0, gain = (
            torch.randn(4, 4096, generator=g).bfloat16(),
            torch.randn(4096, generator=g),
        )
        gain = gain.bfloat16().to(device)
        x0, b0 = parts[rank].to(device), big[rank].to(device)
        x, b, n, r, f, fr = (
            x0.clone(),
            b0.clone(),
            x0.clone(),
            residual0.to(device),
            x0.clone(),
            residual0.to(device),
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            xch.all_reduce(x)
            xch.all_reduce(b)
            xch.add_rms_norm(n, r, gain, 1e-6, True)
            _tile_producer[(1, 32)](xch.epilogue.desc, f, f, 4, 4096, TM=16, TN=128, RANK=rank,
                                    WORLD=WORLD)  # fmt: skip
            xch.reduce(f, tile=(16, 128), residual=fr, norm=(gain, 1e-6, True))
        want, want_big = _rank_sum(parts).bfloat16(), _rank_sum(big).bfloat16()
        ref_x, ref_r = reference(torch.stack(parts), residual0, gain.cpu(), 1e-6)
        doubled = torch.stack([p.float().mul(2).bfloat16() for p in parts])
        ref_fx, ref_fr = reference(doubled, residual0, gain.cpu(), 1e-6)
        for step in range(3):
            for t, src in ((x, x0), (b, b0), (n, x0), (f, x0)):
                t.copy_(src)
            r.copy_(residual0)
            fr.copy_(residual0)
            graph.replay()
            torch.cuda.synchronize()
            expect(f"graph all_reduce {step}", torch.equal(x.cpu(), want))
            expect(f"graph sliced {step}", torch.equal(b.cpu(), want_big))
            expect(f"graph add_rms_norm residual {step}", torch.equal(r.cpu(), ref_r))
            expect(
                f"graph add_rms_norm x {step}",
                torch.allclose(n.cpu().float(), ref_x.float(), rtol=1e-2, atol=1e-2),
            )
            expect(f"graph fused residual {step}", torch.equal(fr.cpu(), ref_fr))
            expect(
                f"graph fused x {step}",
                torch.allclose(f.cpu().float(), ref_fx.float(), rtol=1e-2, atol=1e-2),
            )
            y = x0.clone()
            xch.all_reduce(y)
            expect(f"eager after replay {step}", torch.equal(y.cpu(), want))
        xch.check()

        # Calls above nccl_above go through the communicator (NCCL needs a GPU per rank).
        if torch.cuda.device_count() >= WORLD:
            from ctikernels import init_pynccl

            comm = init_pynccl(tp_rank=rank, tp_size=WORLD, tp_cpu_group=dist.group.WORLD,
                               max_size_bytes=256)  # fmt: skip
            routed = Exchange(
                rank=rank, world_size=WORLD, cpu_group=dist.group.WORLD, device=device,
                max_tokens=64, max_hidden=4096, transport=transport, communicator=comm,
                nccl_above=65536,
            )  # fmt: skip
            parts = _parts(rank, 64, 4096, 55)
            x = parts[rank].to(device)
            routed.all_reduce(x)
            expect("routed all_reduce", torch.equal(x.cpu(), _rank_sum(parts).bfloat16()))
            out = parts[rank].to(device)
            routed.reduce(out, tile=(16, 64))
            expect("routed reduce", torch.equal(out.cpu(), _rank_sum(parts).bfloat16()))
            g = torch.Generator().manual_seed(56)
            residual = torch.randn(64, 4096, generator=g).bfloat16()
            weight = torch.randn(4096, generator=g).bfloat16()
            x, r = parts[rank].to(device), residual.to(device)
            routed.add_rms_norm(x, r, weight.to(device), 1e-6, True)
            ref_x, ref_r = reference(torch.stack(parts), residual, weight, 1e-6)
            # NCCL rounds the sum to BF16 before the residual add (today's composition).
            expect("routed add_rms_norm residual",
                   torch.allclose(r.cpu().float(), ref_r.float(), rtol=1e-2, atol=2e-2))  # fmt: skip
            expect("routed add_rms_norm x",
                   torch.allclose(x.cpu().float(), ref_x.float(), rtol=2e-2, atol=2e-2))  # fmt: skip
            expect("routed add_rms_norm ranks", _same_on_all_ranks(x.cpu()))
            dist.barrier()
            routed.close()

        # Abort: rank 0 calls alone; rank 1 aborts; rank 0's spin ends and check() raises.
        dist.barrier()
        if rank == 0:
            xch.add_rms_norm(x0[:1].clone(), x0[:1].clone(), gain, 1e-6, True)
            try:
                xch.check()
            except RuntimeError:
                pass
            else:
                errors.append("abort did not stop the spin")
        else:
            time.sleep(1.0)
            xch.abort()
        dist.barrier()
        xch.close()
        xch.close()
    except Exception:  # noqa: BLE001  (reported to the parent)
        errors.append(traceback.format_exc())
    queue.put((rank, errors))


@pytest.mark.parametrize("transport", ["host", "peer"])
def test_exchange(transport):
    count = torch.cuda.device_count()
    if transport == "peer" and count > 1 and not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("the GPUs have no peer access")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, port, transport, queue)) for r in range(WORLD)]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, errors = queue.get(timeout=600)
            results[rank] = errors
    except std_queue.Empty:
        pytest.fail(f"exchange workers timed out: {results}")
    finally:
        for p in procs:
            p.join(timeout=60)
    failures = [f"rank {r}: {e}" for r, errors in sorted(results.items()) for e in errors]
    assert not failures, "\n".join(failures)
