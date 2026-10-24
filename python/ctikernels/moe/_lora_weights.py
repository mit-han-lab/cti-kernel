# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Factor storage and metadata-only validation."""

from dataclasses import dataclass

import torch


def validate_rank(rank: int) -> int:
    """Shape check only; each provider states its own rank range."""
    if type(rank) is not int or rank < 1:
        raise ValueError("rank must be a positive integer")
    return rank


def select_adapter(tensor, ndim, adapter_id):
    if type(adapter_id) is not int or adapter_id < 0:
        raise ValueError("adapter_id must be a nonnegative integer")
    if tensor.ndim == ndim:
        if adapter_id != 0:
            raise ValueError("adapter_id must be zero for factors without an adapter dimension")
        return tensor
    if tensor.ndim != ndim + 1 or adapter_id >= tensor.shape[0]:
        raise ValueError("invalid factor dimensions or adapter_id")
    return tensor[adapter_id]


def check_tensor(tensor, name, *, dtype=torch.bfloat16, device=None):
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
        raise ValueError(f"{name} must be a {dtype} tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}")


def check_slices(first, second, name):
    """Gate and up leaves share dtype, device and shape. Kernels address the up leaf at an
    element offset from the gate leaf, so they may be separate allocations or the slices of
    one slice-major tensor; the CUDA provider checks their 16-byte alignment."""
    if first.dtype != second.dtype or first.device != second.device or first.shape != second.shape:
        raise ValueError(f"gate and up {name} must share dtype, device and shape")


class FactorSlices(tuple):
    """One projection's per-slice factors, each [E,R,C]: (gate, up) or (down,).

    ``shape`` is the packed [E,slices,R,C]. Kernels read slice s at ``s * gap`` elements
    from slice zero, so no packed copy exists; torch paths use ``stacked()``.
    """

    @property
    def shape(self):
        return torch.Size((self[0].shape[0], len(self), *self[0].shape[1:]))

    @property
    def gap(self):
        if len(self) == 1:
            return 0
        return (self[1].data_ptr() - self[0].data_ptr()) // self[0].element_size()

    @property
    def requires_grad(self):
        return any(t.requires_grad for t in self)

    def stacked(self):
        return torch.stack(tuple(self), dim=1)

    def packed(self):
        """[E,slices,R,C], a view when every slice lives in one storage, else a stacked copy."""
        first = self[0]
        if len(self) == 1:
            return first.unsqueeze(1)
        if all(t.untyped_storage().data_ptr() == first.untyped_storage().data_ptr() for t in self):
            return first.as_strided(self.shape, (first.stride(0), self.gap, *first.stride()[1:]))
        return self.stacked()


@dataclass(frozen=True)
class MoELoRAWeights:
    """Contiguous BF16 factor leaves; an optional leading adapter axis L is shared by all factors.

    Routed gate/up layout (``layout``):
    - ``"shared_a"``: ``MoELoRAWeights(gate_up_a, gate_up_b, down_a, down_b)``, PEFT/Unsloth's
      ``gate_up_proj``: one A for gate and up, gate_up_a [E,r,H]; gate_up_b [2,E,I,r] holds
      B_gate then B_up (each expert's [2I,r] B, slice-major). Training and inference.
    - ``"separate"``: ``MoELoRAWeights.separate(gate_a, up_a, gate_b, up_b, down_a, down_b)``,
      per-projection adapters (transformers 4.x per-expert Linears): gate_a/up_a [E,r,H],
      gate_b/up_b [E,I,r], separate leaves. Inference only.
    down_a [E,r,I], down_b [E,H,r]. Shared expert (PEFT's separate gate/up Linears): as13
    [2,r,H], bs13 [2,Is,r], as2 [r,Is], bs2 [H,r]. With L, every shape gains a leading L.
    ``routed`` are the kernels' per-slice views (gate_a, up_a, gate_b, up_b, down_a, down_b)
    in either layout (shared_a: A twice, B's two slices), no copies; ``leaves`` the shared-A
    parameters (gate_up_a, gate_up_b, down_a, down_b). Construction reads no device data.
    """

    gate_up_a: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    gate_up_b: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    down_a: torch.Tensor
    down_b: torch.Tensor
    as13: torch.Tensor | None = None
    bs13: torch.Tensor | None = None
    as2: torch.Tensor | None = None
    bs2: torch.Tensor | None = None
    rank: int | None = None

    @classmethod
    def separate(cls, gate_a, up_a, gate_b, up_b, down_a, down_b, *shared, **options):
        return cls((gate_a, up_a), (gate_b, up_b), down_a, down_b, *shared, **options)

    @property
    def layout(self):
        return "separate" if isinstance(self.gate_up_a, tuple) else "shared_a"

    def __post_init__(self):
        separate = isinstance(self.gate_up_a, tuple)
        if separate != isinstance(self.gate_up_b, tuple) or (
            separate and (len(self.gate_up_a) != 2 or len(self.gate_up_b) != 2)
        ):
            raise ValueError("gate_up_a and gate_up_b must both be tensors or (gate, up) pairs")
        if separate:
            names = ("gate_a", "up_a", "gate_b", "up_b")
            leaves = (*self.gate_up_a, *self.gate_up_b)
        else:
            names, leaves = ("gate_up_a", "gate_up_b"), (self.gate_up_a, self.gate_up_b)
        named = (*zip(names, leaves, strict=True), ("down_a", self.down_a), ("down_b", self.down_b))
        # BF16, or FP32 masters for training (cast to BF16 at each use; inference casts once).
        dtype = getattr(leaves[0], "dtype", None)
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("LoRA factors must be BF16 or FP32 tensors")
        for name, t in named:
            check_tensor(t, name, dtype=dtype, device=leaves[0].device)
        a = leaves[0]
        if a.ndim not in (3, 4):
            raise ValueError("routed factors must have shape [E,...] or [L,E,...]")
        r = validate_rank(a.shape[-2])
        if self.rank is not None and validate_rank(self.rank) != r:
            raise ValueError("rank disagrees with factors")
        object.__setattr__(self, "rank", r)
        prefix, h = a.shape[:-2], a.shape[-1]
        i = leaves[2 if separate else 1].shape[-2]
        if any(d <= 0 for d in prefix) or h <= 0 or i <= 0 or h % 8 or i % 8:
            raise ValueError("expert/adapter counts must be positive; H and I multiples of 8")
        if separate:
            shapes = (prefix + (r, h),) * 2 + (prefix + (i, r),) * 2
        else:
            shapes = (prefix + (r, h), prefix[:-1] + (2,) + prefix[-1:] + (i, r))
        shapes += (prefix + (r, i), prefix + (h, r))
        for (name, t), shape in zip(named, shapes, strict=True):
            if t.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        shared = (self.as13, self.bs13, self.as2, self.bs2)
        if any(t is not None for t in shared):
            if any(t is None for t in shared):
                raise ValueError("provide all four shared factors together")
            sp = prefix[:-1]
            if self.bs13.ndim != len(sp) + 3:
                raise ValueError("shared factors must match the routed adapter layout")
            si = self.bs13.shape[-2]
            if si <= 0 or si % 8:
                raise ValueError("shared intermediate size must be a positive multiple of 8")
            shapes = (sp + (2, r, h), sp + (2, si, r), sp + (r, si), sp + (h, r))
            for name, value, shape in zip(
                ("as13", "bs13", "as2", "bs2"), shared, shapes, strict=True
            ):
                check_tensor(value, name, dtype=self.down_a.dtype, device=self.down_a.device)
                if value.shape != shape:
                    raise ValueError(f"{name} must have shape {shape}")

    gate_a = property(lambda self: self.routed[0])
    up_a = property(lambda self: self.routed[1])
    gate_b = property(lambda self: self.routed[2])
    up_b = property(lambda self: self.routed[3])

    def adapter(self, adapter_id):
        """This adapter's factors (the same layout), selected along the leading L axis."""
        pick = lambda t, d: select_adapter(t, d, adapter_id)
        if self.layout == "separate":
            gate_up_a = tuple(pick(t, 3) for t in self.gate_up_a)
            gate_up_b = tuple(pick(t, 3) for t in self.gate_up_b)
        else:
            gate_up_a, gate_up_b = pick(self.gate_up_a, 3), pick(self.gate_up_b, 4)
        shared = ()
        if self.as13 is not None:
            shared = tuple(pick(t, d) for t, d in zip(self.shared, (3, 3, 2, 2), strict=True))
        down = (pick(self.down_a, 3), pick(self.down_b, 3))
        return MoELoRAWeights(gate_up_a, gate_up_b, *down, *shared, rank=self.rank)

    @property
    def dtype(self):
        return self.down_a.dtype

    def to(self, dtype):
        """The same adapters with every factor cast to ``dtype`` (self when already)."""
        if self.dtype == dtype:
            return self
        cast = lambda t: t.to(dtype)
        a, b = self.gate_up_a, self.gate_up_b
        if self.layout == "separate":
            a, b = tuple(map(cast, a)), tuple(map(cast, b))
        else:
            a, b = cast(a), cast(b)
        shared = () if self.as13 is None else tuple(map(cast, self.shared))
        return MoELoRAWeights(a, b, cast(self.down_a), cast(self.down_b), *shared, rank=self.rank)

    @property
    def leaves(self):
        if self.layout != "shared_a":
            raise ValueError("separate gate/up adapters are inference-only (train shared-A)")
        return (self.gate_up_a, self.gate_up_b, self.down_a, self.down_b)

    @property
    def routed(self):
        if self.layout == "separate":
            return (*self.gate_up_a, *self.gate_up_b, self.down_a, self.down_b)
        return routed_slices(self.gate_up_a, self.gate_up_b, self.down_a, self.down_b)

    @property
    def shared(self):
        return None if self.as13 is None else (self.as13, self.bs13, self.as2, self.bs2)


def routed_slices(gate_up_a, gate_up_b, down_a, down_b):
    """The kernels' per-slice routed factors (A, A, B_gate, B_up, down_a, down_b): A is read
    for both gate and up (slice gap 0), B's slices are views of the slice-major leaf."""
    b = gate_up_b
    b_gate, b_up = (b[0], b[1]) if b.ndim == 4 else (b[:, 0], b[:, 1])
    return gate_up_a, gate_up_a, b_gate, b_up, down_a, down_b
