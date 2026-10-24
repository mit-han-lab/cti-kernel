# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""cuBLASLt static FP8 GEMMs. All host reads/probes happen at construction."""

import threading

import torch

_ROWWISE_PROBES = {}
_PROBE_LOCK = threading.Lock()
OUTPUT_DTYPES = (torch.bfloat16, torch.float16)


def available(k, slot_rows):
    return k % 16 == 0 and all(rows % 16 == 0 for rows in slot_rows)


def _equal_scales(scale):
    """Construction only: never reduce distinct scales to a maximum."""
    return scale.numel() == 1 or bool(torch.all(scale == scale.reshape(-1)[0]).item())


def column_scales(weight_scale, slot_rows):
    """Pure device tensor preparation of contiguous FP32 [1,N], with no host read."""
    return torch.cat(
        [weight_scale.reshape(-1)[i : i + 1].expand(rows) for i, rows in enumerate(slot_rows)]
    ).reshape(1, sum(slot_rows))


def rowwise_probe(device, dtype):
    """Cache None (supported) or the RuntimeError text per concrete device/dtype."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = device, dtype
    with _PROBE_LOCK:
        if key not in _ROWWISE_PROBES:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("rowwise probe must run before capture")
                try:
                    torch._scaled_mm(
                        torch.zeros((16, 32), device=device, dtype=torch.float8_e4m3fn),
                        torch.zeros((16, 32), device=device, dtype=torch.float8_e4m3fn).t(),
                        scale_a=torch.ones((16, 1), device=device, dtype=torch.float32),
                        scale_b=torch.ones((1, 16), device=device, dtype=torch.float32),
                        out_dtype=dtype,
                        use_fast_accum=False,
                    )
                except RuntimeError as error:
                    _ROWWISE_PROBES[key] = str(error)
                else:
                    _ROWWISE_PROBES[key] = None
    return _ROWWISE_PROBES[key]


def _has_out():
    # Builtins need not expose inspect.signature; the ATen schema is authoritative.
    return "out" in torch.ops.aten._scaled_mm.overloads()


def aligned(x):
    x = x.contiguous()
    if x.data_ptr() % 16 or x.stride(0) % 16 or x.stride(0) < x.shape[1]:
        x = x.clone(memory_format=torch.contiguous_format)
    return x


def direct_output(out):
    return out.stride(0) == out.shape[1] and out.is_contiguous() and out.data_ptr() % 16 == 0


class TorchScaledMM:
    def __init__(self, weight, weight_scale, input_scale, slot_rows, mode="slots"):
        equal = _equal_scales(weight_scale)
        rows = (weight.shape[0],) if equal else slot_rows
        if weight.shape[1] % 16 or weight.shape[0] % 16:
            raise ValueError("torch-scaled-mm needs K and N divisible by 16")
        self.providers = {}
        self.errors = {}
        for dtype in OUTPUT_DTYPES:
            error = (
                rowwise_probe(weight.device, dtype)
                if mode == "rowwise" or (mode == "auto" and not equal)
                else None
            )
            chosen = mode
            if mode == "auto":
                chosen = "slots" if equal or error is not None else "rowwise"
            if chosen == "rowwise" and dtype == torch.float16:
                # Measured on RTX 5090 / RTX PRO 6000 / GB10 (torch 2.13): rowwise FP16
                # outputs miss the FP32-floor precision gate that BF16 rowwise and the
                # per-slot tensorwise calls pass, so FP16 always takes the per-slot form.
                chosen, error = "slots", None
            if chosen == "rowwise" and error is not None:
                self.errors[dtype] = error
            elif chosen == "slots" and not available(weight.shape[1], rows):
                self.errors[dtype] = "torch-scaled-mm needs K and every slot's rows divisible by 16"
            self.providers[dtype] = "torch-scaled-mm:" + chosen
        self._input_scale = input_scale.reshape(1)
        self._has_out = _has_out()
        self._weight = aligned(weight).t()
        self._columns = column_scales(weight_scale, slot_rows)
        self._slots = []
        scales = weight_scale.reshape(-1)
        start = 0
        for i, count in enumerate(rows):
            stop = start + count
            self._slots.append((slice(start, stop), self._weight[:, start:stop], scales[i : i + 1]))
            start = stop

    def run(self, q, out, row_scale=None, *, buffers=None):
        if buffers is None:
            q = aligned(q)
        elif not direct_output(q) or q.stride(0) % 16:
            q = buffers.get("q").copy_(q)
        rowwise = self.providers[out.dtype].endswith(":rowwise")
        if rowwise:
            # Prequantized entry needs one extra small copy kernel for M > 1.
            if row_scale is None:
                source = self._input_scale.reshape(1, 1).expand(q.shape[0], 1)
                row_scale = (
                    source.contiguous() if buffers is None else buffers.get("scale").copy_(source)
                )
            slots = ((slice(0, out.shape[1]), self._weight, self._columns),)
        else:
            slots = self._slots
        for columns, weight, scale in slots:
            kwargs = {
                "scale_a": row_scale if rowwise else self._input_scale,
                "scale_b": scale,
                "out_dtype": out.dtype,
                "use_fast_accum": False,
            }
            if len(slots) == 1 and self._has_out and direct_output(out):
                torch._scaled_mm(q, weight, out=out, **kwargs)
            elif buffers is not None:
                target = (
                    buffers.get("staging", out.dtype)
                    .view(-1)[: q.shape[0] * weight.shape[1]]
                    .view(q.shape[0], weight.shape[1])
                )
                torch._scaled_mm(q, weight, out=target, **kwargs)
                out[:, columns].copy_(target)
            else:
                out[:, columns].copy_(torch._scaled_mm(q, weight, **kwargs))
