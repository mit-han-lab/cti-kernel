# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Independent test composition of engine FR and its D call site (R5 §5.1).

FR's tail fill and scratch writes are retained deliberately. This composition
is only a legal-input oracle; extreme values use the index-free scalar oracle.
Also used as the torch graph benchmark baseline.
"""

import torch

from ctikernels.speculative import (
    commit_block,
    fixed_query_addresses,
    prepare_block,
    prepare_round,
)
from ctikernels.speculative._round.op import A_NAMES, C_NAMES, P_NAMES, R_NAMES

SENTINEL = 0x7FFFFFFF
MODES = ("greedy", "prefilled", "sampled")
BATCHES = (1, 2, 3, 4, 5, 8, 16, 32, 48, 64, 191, 256, 1024)
ANY_QS = (1, 2, 4, 5, 8, 9, 12, 16, 33, 64)
QS = (16, 8, 2, 4, 5, 6, 9, 12, 31, 32)


def fixture(b=4, q=8, m=67, *, mode="greedy", s=3, turn=0, optional=True):
    if b < 0:
        raise ValueError("batch size must be nonnegative")
    n_state = 9 if b <= 4 else b + 5
    wt, wc = ((m + q + 127) // 128) * 128, ((m + 63) // 64) * 64
    i32, i64 = torch.int32, torch.int64
    row_order = [5, 0, 3, 7] if b <= 4 else [n_state - 1, 0, *range(n_state - 2, 0, -1)]
    rows = torch.tensor(row_order[:b], dtype=i32)
    fronts = torch.zeros(n_state, dtype=i64)
    cases = (0, 1, m - q - 1, m - q, m - q + 1, m - 2, m - 1)
    fronts[rows.long()] = torch.tensor([cases[(turn + i) % 7] for i in range(b)], dtype=i64)
    done = torch.zeros(n_state, dtype=torch.bool)
    if b > 4:
        done[rows.long()] = (torch.arange(b) + turn) % 3 == 1
    elif turn % (b + 2) == b + 1:
        done[rows.long()] = True
    elif turn % (b + 2) > 0:
        done[rows[turn % (b + 2) - 1].long()] = True
    table = torch.arange((n_state + 1) * wt, dtype=i32).view(n_state + 1, wt) + 1000
    cache = torch.arange((n_state + 2) * wc, dtype=i32).view(n_state + 2, wc) * 3 + 10
    padding = 8 if b <= 4 else n_state
    table[padding:].zero_()
    cache[padding:].zero_()
    targets = (torch.arange(b * q, dtype=i64) + 100 + turn * 100).view(b, q)
    # Engine view: unit inner stride, row stride Q instead of Q-1.
    proposal_backing = torch.zeros(b, q, dtype=i32)
    proposals = proposal_backing[:, 1:]
    proposals.copy_(targets[:, :-1])
    accepted = torch.tensor([(turn + i) % q for i in range(b)], dtype=i32)
    for i, a in enumerate(accepted.tolist()):
        if a < q - 1:
            proposals[i, a] += 1
    result = {
        "request_indices": rows,
        "frontier_table": fronts,
        "done_table": done,
        "token_table": table,
        "cache_index_table": cache,
        "start_frontiers": torch.zeros(b, dtype=i64),
        "round_done": torch.zeros(b, dtype=torch.bool),
        "input_ids": torch.zeros(b * q, dtype=i32),
        "positions": torch.zeros(b * q, dtype=i64),
        "rope_positions": torch.zeros(b * q, dtype=i64),
        "cache_write_indices": torch.zeros(b * q, dtype=i32),
        "kv_indptr": torch.zeros(b + 1, dtype=i32),
        "kv_indices": torch.full((b * (m + q),), SENTINEL, dtype=i32),
        "tokens": targets.to(i32),
        "accepted": accepted,
        "output_ends": torch.full((b,), m, dtype=i64),
        "ignore_eos": torch.tensor([i % 2 for i in range(b)], dtype=i32),
        "stop_token_ids": torch.arange(s, dtype=i32)
        + (int(targets[0, turn % q]) if b else 100 + turn * 100 + turn % q),
        "committed": torch.zeros(b, dtype=i32),
        "record_done": torch.zeros(b, dtype=i32),
        "proposals": proposals if mode != "sampled" else None,
        "target_ids": targets.flatten() if mode == "greedy" else None,
        "rope_offsets": torch.arange(b, dtype=i64) + 2 if optional else None,
        "prompt_lengths": torch.arange(b, dtype=i64) + 3 if optional else None,
        "k2_slots": torch.zeros(b, dtype=i32) if optional else None,
        "k2_lengths": torch.zeros(b, dtype=i32) if optional else None,
        "verify_rope_positions": torch.zeros(b * q, dtype=i64) if optional else None,
        "cursors": torch.zeros(b, dtype=i64) if optional else None,
        "block_size": q,
        "max_seq_len": m,
        "guard_start": 10000000,
        "mask_token_id": 17,
        "use_pdl": False,
    }
    return result


def clone(t, device="cpu"):
    result = {}
    for n, v in t.items():
        if isinstance(v, torch.Tensor):
            # Preserve strides and alignment of sliced views. Initializing the
            # whole backing also makes sanitizer initcheck independent of gaps.
            offset = v.storage_offset() % 4
            span = (
                0
                if not v.numel()
                else 1
                + sum((size - 1) * stride for size, stride in zip(v.shape, v.stride(), strict=True))
            )
            backing = torch.zeros(span + offset, dtype=v.dtype, device=device)
            out = backing.as_strided(v.shape, v.stride(), offset)
            out.copy_(v)
            result[n] = out
        else:
            result[n] = v
    return result


def prepare(t, split=False):
    if split:
        prepare_split(t)
        return
    prepare_block(
        **{n: t[n] for n in P_NAMES},
        **{
            n: t[n]
            for n in ("block_size", "max_seq_len", "guard_start", "mask_token_id", "use_pdl")
        },
    )


def prepare_split(t):
    """SP7: prepare_round, then fixed_query_addresses on its saved state."""
    prepare_round(
        **{n: t[n] for n in R_NAMES},
        **{n: t[n] for n in ("block_size", "max_seq_len", "mask_token_id", "use_pdl")},
    )
    fixed_query_addresses(
        **{n: t[n] for n in A_NAMES},
        **{n: t[n] for n in ("block_size", "max_seq_len", "guard_start", "use_pdl")},
    )


def commit(t):
    commit_block(
        **{n: t[n] for n in C_NAMES}, **{n: t[n] for n in ("block_size", "max_seq_len", "use_pdl")}
    )


def round_call(t, split=False):
    prepare(t, split)
    commit(t)


def fr_prepare(t):
    q, m, g = t["block_size"], t["max_seq_len"], t["guard_start"]
    rows = t["request_indices"].long()
    b = rows.numel()
    if not b:
        return
    f, d = t["frontier_table"][rows], t["done_table"][rows]
    t["start_frontiers"].copy_(f)
    t["round_done"].copy_(d)
    if t["k2_slots"] is not None:
        t["k2_slots"].copy_(torch.where(d, -1, t["request_indices"]))
    i = torch.arange(q, device=rows.device)
    guard = g + torch.arange(b, device=rows.device)[:, None] * q
    logical = f[:, None] + i
    rope = logical.clamp(max=m - 1)
    t["positions"].copy_(logical.flatten())
    t["rope_positions"].copy_(rope.flatten())
    writes = torch.where(
        logical < m,
        t["cache_index_table"][rows[:, None], rope],
        guard + (logical - m).clamp(0, q - 1),
    )
    t["cache_write_indices"].copy_(torch.where(d[:, None], guard + i, writes).flatten())
    ids = t["input_ids"].view(b, q)
    ids.fill_(t["mask_token_id"])
    ids[:, 0] = t["token_table"][rows, f.clamp(max=m - 1)]
    lengths = torch.where(d, q, f + q)
    ends = lengths.cumsum(0)
    t["kv_indptr"][:1].zero_()
    t["kv_indptr"][1:].copy_(ends)
    flat = torch.arange(t["kv_indices"].numel(), device=rows.device)
    row = torch.searchsorted(ends, flat, right=True).clamp(max=b - 1)
    local = flat - (ends[row] - lengths[row])
    row_guard = guard[row, 0]
    keys = torch.where(
        local < m,
        t["cache_index_table"][rows[row], local.clamp(0, m - 1)],
        row_guard + (local - m).clamp(0, q - 1),
    )
    keys = torch.where(d[row], row_guard + local.clamp(0, q - 1), keys)
    # FR:97 is intentionally preserved; compare only its valid prefix.
    t["kv_indices"].copy_(torch.where(flat < ends[-1], keys, g))
    if t["verify_rope_positions"] is not None:
        t["verify_rope_positions"].copy_((rope + t["rope_offsets"][:, None]).flatten())
    if t["cursors"] is not None:
        t["cursors"].copy_(f - t["prompt_lengths"] + 1)


def fr_commit(t):
    q, m = t["block_size"], t["max_seq_len"]
    rows = t["request_indices"].long()
    if not rows.numel():
        return
    f, d = t["start_frontiers"], t["round_done"]
    tokens = t["tokens"]
    if t["target_ids"] is not None:
        tokens.copy_(t["target_ids"].view(-1, q))
    if t["proposals"] is not None:
        t["accepted"].copy_(
            (t["proposals"] == tokens[:, :-1]).int().cumprod(1).sum(1, dtype=torch.int32)
        )
    e = t["output_ends"].clamp(max=m)
    limit = torch.minimum((e - 1 - f).clamp(0, q), t["accepted"].long() + 1)
    i = torch.arange(q, device=rows.device)
    stops = (tokens[..., None] == t["stop_token_ids"]).any(-1)
    stops &= ~(t["ignore_eos"] != 0)[:, None] & (i < limit[:, None])
    first = torch.where(stops, i, q).amin(1)
    v = torch.where(d, 0, torch.where(first < q, first + 1, limit))
    fin = d | (first < q) | (f + v >= e - 1)
    columns = torch.where(i < v[:, None], f[:, None] + 1 + i, m + i)
    t["token_table"][rows[:, None], columns] = tokens
    t["frontier_table"][rows], t["done_table"][rows] = f + v, fin
    t["accepted"].masked_fill_(d, -1)
    t["committed"].copy_(v)
    t["record_done"].copy_(fin)
    if t["k2_lengths"] is not None:
        t["k2_lengths"].copy_(v)


def assert_equal(actual, expected, *, fr=False):
    for name, v in actual.items():
        if not isinstance(v, torch.Tensor):
            continue
        x, y = v.cpu(), expected[name].cpu()
        if name == "kv_indices":
            end = min(int(actual["kv_indptr"][-1]), v.numel())
            torch.testing.assert_close(x[:end], y[:end], rtol=0, atol=0, msg=name)
            assert (x[end:] == SENTINEL).all(), "kv_indices tail modified"
        elif name == "token_table" and fr:
            y = y.clone()
            m, q = actual["max_seq_len"], actual["block_size"]
            rows = actual["request_indices"].cpu().long()
            y[rows, m : m + q] = x[rows, m : m + q]
            torch.testing.assert_close(x, y, rtol=0, atol=0, msg=name)
        else:
            torch.testing.assert_close(x, y, rtol=0, atol=0, msg=name)


def semantic_cases(b, q, mode, *, full=True):
    """Cover each semantic axis without an unbounded Cartesian product."""
    if b == 0:
        yield fixture(b, q, 67, mode=mode)
        return
    for turn in range(max(q, 7)):
        t = fixture(b, q, 67, mode=mode, turn=turn, s=(0, 1, 3)[turn % 3])
        f = t["frontier_table"][t["request_indices"].long()]
        t["output_ends"].copy_(f + 1 + (-3, 0, 1, q - 1, q, q + 1, 10**9)[turn % 7])
        yield t
    # Each a, each stop position, two stops, ignored stops, and an EOS anchor.
    for a in range(q):
        for scenario in range(6 if full else 2):
            t = fixture(b, q, 67, mode=mode, s=(0 if scenario == 0 else 3))
            rows = t["request_indices"].long()
            t["frontier_table"][rows] = 1
            t["done_table"].zero_()
            t["accepted"].fill_(a)
            if mode != "sampled":
                t["proposals"].copy_(t["tokens"][:, :-1])
                if a < q - 1:
                    t["proposals"][:, a] += 7
            t["ignore_eos"].fill_(-3 if scenario == 5 else 0)
            if scenario:
                # All rows have the same token IDs, so every row exercises stops.
                t["tokens"].copy_(t["tokens"][:1].expand(b, q).clone())
                if t["target_ids"] is not None:
                    t["target_ids"].copy_(t["tokens"].flatten())
                if t["proposals"] is not None:
                    t["proposals"].copy_(t["tokens"][:, :-1])
                    if a < q - 1:
                        t["proposals"][:, a] += 7
                stop_i = a if scenario in (1, 2, 5) else min(a + 1, q - 1)
                t["stop_token_ids"][:] = torch.tensor([int(t["tokens"][0, stop_i]), -400, -401])
                if scenario == 2:
                    t["stop_token_ids"][1] = t["tokens"][0, 0]
                if scenario == 4:
                    t["output_ends"].fill_(3)  # one-token budget; later stop ignored
                t["token_table"][rows, 1] = t["stop_token_ids"][0]
            yield t
    # Every frontier and done pattern independently, including every single done row.
    for f in (0, 1, 67 - q - 1, 67 - q, 67 - q + 1, 65, 66):
        for done in range(b + 2):
            t = fixture(b, q, 67, mode=mode, s=0)
            rows = t["request_indices"].long()
            t["frontier_table"][rows] = f
            t["done_table"].zero_()
            if done == b + 1:
                t["done_table"][rows] = True
            elif done:
                t["done_table"][rows[done - 1]] = True
            yield t
