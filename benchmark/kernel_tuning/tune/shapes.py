# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Projection and MoE shapes of the target models, per TP/EP shard.

Values come from the HF configs (Qwen3.5-9B, Qwen3.5-27B, Qwen3.5-35B-A3B-FP8) and
DeepSeek-V4-Flash at the revision under artifacts/dsv4. Attention q carries the output
gate (2 x heads x head_dim); GDN in_proj_qkv = 2 x key heads x 128 + value heads x 128.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    gdn_key_heads: int
    gdn_value_heads: int
    intermediate: int = 0  # dense MLP
    experts: int = 0
    top_k: int = 0
    expert_intermediate: int = 0
    shared_intermediate: int = 0


MODELS = {
    "9b": Model(4096, 16, 4, 256, 16, 32, intermediate=12288),
    "27b": Model(5120, 24, 4, 256, 16, 48, intermediate=17408),
    "35b-a3b": Model(2048, 16, 2, 256, 16, 32, 0, 256, 8, 512, 512),
    "dsv4-flash": Model(4096, 64, 1, 512, 0, 0, 0, 256, 6, 2048, 2048),
}


@dataclass(frozen=True)
class Projection:
    name: str
    k: int  # input features of this shard
    sizes: tuple[int, ...]  # output slice widths of this shard


def projections(model: str, tp: int) -> list[Projection]:
    """LoRA-carrying dense projections of one layer, as one TP rank sees them.

    Column-parallel ones split their output slices, row-parallel ones their input.
    """
    m = MODELS[model]
    result = []
    if m.gdn_key_heads:
        key, value = m.gdn_key_heads * 128, m.gdn_value_heads * 128
        q = 2 * m.heads * m.head_dim
        kv = m.kv_heads * m.head_dim
        result += [
            Projection(
                "attn_qkv",
                m.hidden,
                (q // tp, max(kv // tp, m.head_dim), max(kv // tp, m.head_dim)),
            ),
            Projection("attn_o", m.heads * m.head_dim // tp, (m.hidden,)),
            Projection("gdn_qkv", m.hidden, (key // tp, key // tp, value // tp)),
            Projection("gdn_z", m.hidden, (value // tp,)),
            Projection("gdn_out", value // tp, (m.hidden,)),
        ]
    if m.intermediate:
        result += [
            Projection("mlp_gate_up", m.hidden, (m.intermediate // tp, m.intermediate // tp)),
            Projection("mlp_down", m.intermediate // tp, (m.hidden,)),
        ]
    if m.shared_intermediate:
        i = m.shared_intermediate
        result += [
            Projection("shared_gate_up", m.hidden, (i // tp, i // tp)),
            Projection("shared_down", i // tp, (m.hidden,)),
        ]
    return result


@dataclass(frozen=True)
class Experts:
    e_local: int
    top_k: int
    hidden: int
    intermediate: int


def experts(model: str, ep: int) -> Experts:
    m = MODELS[model]
    if not m.experts:
        raise ValueError(f"{model} has no routed experts")
    return Experts(m.experts // ep, m.top_k, m.hidden, m.expert_intermediate)
