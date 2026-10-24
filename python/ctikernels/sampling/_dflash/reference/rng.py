# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Integer-only implementation of the published Philox-4x32-10 algorithm.

This is an independent implementation of the algorithm, not copied source.
The oracle accepts a uint64 seed integer; device storage contains its two uint32 keys.
"""

K1_RNG_ABI_VERSION = 1
PROPOSAL, ACCEPT, CORRECTION, BONUS = 1, 2, 3, 4
ORDINARY_NAMESPACE = 0x100
ORDINARY_NAMESPACE_END = 0x1FF
MASK = 0xFFFFFFFF


def philox4x32(counter, key):
    """Return four uint32 words; arguments contain four/two uint32 integers."""
    if len(counter) != 4 or len(key) != 2:
        raise ValueError("Philox needs four counter words and two key words")
    if any(not 0 <= x <= MASK for x in (*counter, *key)):
        raise ValueError("Philox words must fit uint32")
    a, b, c, d = counter
    lo, hi = key
    for _ in range(10):
        ac, cc = a * 0xD2511F53, c * 0xCD9E8D57
        a, b, c, d = (cc >> 32) ^ b ^ lo, cc & MASK, (ac >> 32) ^ d ^ hi, ac & MASK
        lo, hi = (lo + 0x9E3779B9) & MASK, (hi + 0xBB67AE85) & MASK
    return a, b, c, d


def words(seed, cursor, domain, i):
    if not 0 <= seed < 2**64 or not 0 <= cursor < 2**63:
        raise ValueError("seed must fit uint64 and cursor must be nonnegative int64")
    return philox4x32((cursor & MASK, cursor >> 32, domain, i), (seed & MASK, seed >> 32))


def uniform(seed, cursor, domain, i):
    """Exactly representable FP32 uniform in [0, 1), with 24 random bits."""
    return (words(seed, cursor, domain, i)[0] >> 8) * 2.0**-24


__all__ = [
    "K1_RNG_ABI_VERSION",
    "PROPOSAL",
    "ACCEPT",
    "CORRECTION",
    "BONUS",
    "ORDINARY_NAMESPACE",
    "ORDINARY_NAMESPACE_END",
    "MASK",
    "philox4x32",
    "words",
    "uniform",
]
