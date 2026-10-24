# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""MoE LoRA test factors: gate/up leaves in slice-major storage, also seen as [..., E, 2, ...]."""

from ctikernels.moe.lora import MoELoRAWeights


def pair_view(gate, up):
    """[..., E, 2, R, C] view of two leaves of one storage (no copy, writes alias)."""
    gap = (up.data_ptr() - gate.data_ptr()) // gate.element_size()
    shape = (*gate.shape[:-2], 2, *gate.shape[-2:])
    return gate.as_strided(shape, (*gate.stride()[:-2], gap, *gate.stride()[-2:]))


class Packed(MoELoRAWeights):
    """MoELoRAWeights whose reference math reads packed gate/up views."""

    a13 = property(lambda self: pair_view(self.gate_a, self.up_a))
    b13 = property(lambda self: pair_view(self.gate_b, self.up_b))
    a2 = property(lambda self: self.down_a)
    b2 = property(lambda self: self.down_b)


def packed(a13, b13, a2, b2, *shared, **options):
    """Weights from [..., E, 2, ...] gate/up factors (one slice-major copy; differentiable)."""
    sa, sb = (t.movedim(-3, 0).contiguous() for t in (a13, b13))
    return Packed.separate(sa[0], sa[1], sb[0], sb[1], a2, b2, *shared, **options)


def shared_a(a, b13, a2, b2, *shared, **options):
    """Shared-A weights from one A [..., E, r, H] and packed [..., E, 2, I, r] B (differentiable)."""
    return MoELoRAWeights(a, b13.movedim(-3, -4).contiguous(), a2, b2, *shared, **options)


def separate(weights):
    """The same values with every routed leaf in its own allocation."""
    return MoELoRAWeights.separate(*(t.clone() for t in weights.routed), *(weights.shared or ()))


def leaves13(a13, b13):
    """(gate_a, up_a, gate_b, up_b) copies of packed [..., E, 2, ...] factors (differentiable)."""
    return tuple(t.select(-3, s).contiguous() for t in (a13, b13) for s in (0, 1))
