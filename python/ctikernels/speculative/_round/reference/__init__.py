# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only, torch storage reference with explicit arbitrary-integer arithmetic."""

import torch


def wrap64(x):
    return (x + (1 << 63)) % (1 << 64) - (1 << 63)


class Provider:
    def __init__(self, device):
        self.device = device

    def warm_up(self, q, s, use_pdl, variants):
        pass

    def prepare(self, t):
        self.prepare_round(t)
        self.addresses(t)

    def prepare_round(self, t):
        q, m = t["block_size"], t["max_seq_len"]
        for b, r in enumerate(t["request_indices"].tolist()):
            f, d = int(t["frontier_table"][r]), bool(t["done_table"][r])
            t["start_frontiers"][b], t["round_done"][b] = f, d
            if t["k2_slots"] is not None:
                t["k2_slots"][b] = -1 if d else r
            if t["cursors"] is not None:
                t["cursors"][b] = wrap64(f - int(t["prompt_lengths"][b]) + 1)
            for i in range(q):
                k = b * q + i
                p = wrap64(f + i)
                rope = min(p, m - 1)
                t["positions"][k], t["rope_positions"][k] = p, rope
                if t["verify_rope_positions"] is not None:
                    t["verify_rope_positions"][k] = wrap64(rope + int(t["rope_offsets"][b]))
                t["input_ids"][k] = (
                    t["token_table"][r, min(max(f, 0), m - 1)] if i == 0 else t["mask_token_id"]
                )

    def addresses(self, t):
        # Reads only the saved round state (start_frontiers, round_done).
        q, m, offset = t["block_size"], t["max_seq_len"], 0
        t["kv_indptr"][0] = 0
        for b, r in enumerate(t["request_indices"].tolist()):
            f, d = int(t["start_frontiers"][b]), bool(t["round_done"][b])
            dummy = d or not 0 <= f < m
            g = t["guard_start"] + b * q
            for i in range(q):
                t["cache_write_indices"][b * q + i] = (
                    g + i
                    if dummy
                    else t["cache_index_table"][r, f + i]
                    if f + i < m
                    else g + min(f + i - m, q - 1)
                )
            length = q if dummy else f + q
            count = max(0, min(length, t["kv_indices"].numel() - offset))
            # Work proportional to the actual prefix, including short N.
            if dummy:
                t["kv_indices"][offset : offset + count] = torch.arange(
                    g, g + count, dtype=torch.int32
                )
            else:
                mapped = min(count, m)
                t["kv_indices"][offset : offset + mapped] = t["cache_index_table"][r, :mapped]
                if count > mapped:
                    t["kv_indices"][offset + mapped : offset + count] = torch.arange(
                        g, g + count - mapped, dtype=torch.int32
                    )
            offset += length
            t["kv_indptr"][b + 1] = offset

    def commit(self, t):
        q, m = t["block_size"], t["max_seq_len"]
        if t["target_ids"] is not None:
            t["tokens"].copy_(t["target_ids"].view(-1, q))
        stops = set(t["stop_token_ids"].tolist())
        for b, r in enumerate(t["request_indices"].tolist()):
            f, d = int(t["start_frontiers"][b]), bool(t["round_done"][b])
            if t["proposals"] is None:
                a = int(t["accepted"][b])
            else:
                a = 0
                while a < q - 1 and t["proposals"][b, a] == t["tokens"][b, a]:
                    a += 1
            v, fin = 0, True
            if not d and 0 <= f < m:
                e = min(int(t["output_ends"][b]), m)
                budget = 0 if e <= f + 1 else min(e - f - 1, q)
                limit = min(budget, a + 1)
                s = next(
                    (
                        i
                        for i in range(q)
                        if i < limit
                        and int(t["ignore_eos"][b]) == 0
                        and int(t["tokens"][b, i]) in stops
                    ),
                    q,
                )
                v = s + 1 if s < q else limit
                fin = s < q or f + v + 1 >= e
                for i in range(max(0, v)):
                    t["token_table"][r, f + 1 + i] = t["tokens"][b, i]
            t["frontier_table"][r], t["done_table"][r] = f + v, fin
            t["accepted"][b] = -1 if d else a
            t["committed"][b], t["record_done"][b] = v, int(fin)
            if t["k2_lengths"] is not None:
                t["k2_lengths"][b] = v
