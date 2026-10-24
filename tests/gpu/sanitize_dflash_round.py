# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Run with python -m tests.gpu.sanitize_dflash_round under compute-sanitizer.

All addressed tensor elements are initialized before kernel reads; row-stride
gaps are deliberately unread.
Use --full for the full independently sampled semantic matrix.
"""

import argparse
import json
from itertools import product

import torch

from ctikernels.speculative import warm_up

from .._dflash_round_reference import MODES, QS, fixture, semantic_cases
from ..modules.test_dflash_round import extreme_cases
from .test_dflash_round import (
    ADDRESSES,
    addresses_case,
    call_addresses,
    check_addresses,
    check_case,
    test_cross_stream_row_ownership,
    test_d2_delayed_producers_detect_missing_wait,
    to_cuda,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for sanitizer qualification")
    torch.set_num_threads(1)
    count = 0
    for q in QS if args.full else (2, 8, 16, 32):
        for s in (0, 1, 3):
            warm_up(q, 262144, s, device="cuda")
        for b in range(1, 5):
            for mode in MODES:
                for t in semantic_cases(b, q, mode, full=args.full):
                    for split in (False, True):
                        check_case(t, split=split)
                        count += 1
        for m in (64, 4096, 40960, 131072, 262144):
            for turn in range(7):
                for split in (False, True):
                    check_case(fixture(4, q, m, turn=turn, mode=MODES[turn % 3]), split=split)
                    count += 1
    for b in (5, 16, 191):
        for q in (1, 33):
            warm_up(q, 67, 3, device="cuda")
            for mode in MODES:
                for optional in (False, True):
                    for split in (False, True):
                        t = fixture(b, q, 67, mode=mode, optional=optional)
                        t["frontier_table"].clamp_(max=66)
                        check_case(t, split=split)
                        count += 1
    for t in extreme_cases():
        for split in (False, True):
            check_case(t, fr=False, split=split)
            count += 1
    for a in (-(2**31), -2, 8, 2**31 - 1):
        t = fixture(mode="sampled", s=0)
        t["frontier_table"].fill_(1)
        t["accepted"].fill_(a)
        check_case(t, fr=False)
        count += 1
    for n in (0, 1, 7, 32):
        t = fixture(mode="sampled", s=0)
        t["kv_indices"] = t["kv_indices"][:n]
        for split in (False, True):
            check_case(t, fr=False, split=split)
            count += 1
    # Triton fixed_query_addresses: boundary frontiers, guards, short capacities.
    for q in (1, 5, 16, 33):
        warm_up(q, 67, 0, device="cuda", variants=ADDRESSES)
        for m, b, mix in product(sorted({1, q, 2 * q + 1, 67}), (1, 4, 5, 64), range(6)):
            for capacity in (None, 0, 1, 2 * q + 1):
                t = addresses_case(
                    b, q, m, mix=mix, guard_max=mix % 2 == 1, capacity=capacity, shift=mix % 2
                )
                actual = to_cuda(t)
                call_addresses(actual)
                check_addresses(actual, t)
                count += 1
    # Sanitizers may serialize dependent launches. Exercise the correct D2 and
    # cross-stream paths; mutation sensitivity is qualified without instrumentation.
    for b in (4, 8):
        for mode in MODES:
            for captured in (False, True):
                for split in (False, True):
                    test_d2_delayed_producers_detect_missing_wait(
                        mode, captured, split, b, detect_mutants=False
                    )
        for split in (False, True):
            test_cross_stream_row_ownership(False, split, b)
    torch.cuda.synchronize()
    print(json.dumps({"cases": count, "ok": True}))


if __name__ == "__main__":
    main()
