# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Independent host construction of DF4's logical key lists."""

import torch


def table_csr(requests, frontier, done, table, guard_start, maximum, query):
    ptr, keys, positions = [0], [], []
    for b, r in enumerate(requests.tolist()):
        f = int(frontier[r])
        dummy = bool(done[r]) or not 0 <= f < maximum
        length = query if dummy else f + query
        guard = guard_start + b * query
        for j in range(length):
            keys.append(
                guard + j
                if dummy
                else int(table[r, j])
                if j < maximum
                else guard + min(j - maximum, query - 1)
            )
        ptr.append(len(keys))
        positions.extend(range(length - query, length))
    return (
        torch.tensor(ptr, dtype=torch.int32),
        torch.tensor(keys, dtype=torch.int32),
        torch.tensor(positions, dtype=torch.int64),
    )
