# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Frozen GEN-GDN baseline from b4dce8a2; do not update to match dispatch changes."""


def legacy_config(batch: int, hv: int, hk: int, sms: int, pro: bool) -> tuple[int, int, bool]:
    """(state rows per CTA, warps, finishers); 96 is the three-CTA split (64 + 32 + 32 rows).

    Default rules (RTX 5090; GB10 at its SM count), heads = batch * Hv, R = Hv / Hk:
    - R = 3, three CTAs per head + one finisher per key head fit one per SM and four would not:
      the split;
    - 4 heads <= SMs: quarters (32/4) with finishers for R <= 4;
    - heads <= SMs: the split where 2 heads overflow the SMs by 4-13% (R = 2..4), else halves
      (64/4) with finishers for R = 3, 4 while 3 heads <= SMs, flags otherwise;
    - beyond: 64/4 with finishers while the grid fits three CTAs per SM (R <= 4; and for R = 3,
      Hk >= 8 while 2 heads <= 3 SMs), 64/4 flags while 2 heads <= 3 SMs or 5 SMs < 2 heads <=
      6 SMs, whole heads (128/8) from 2.15 heads per SM, else 64/8 flags.
    RTX PRO 6000: quarters with finishers while 4 heads <= SMs; up to one head per SM whole
    heads once 2 heads > SMs + 4, else 64/4 flags; beyond, 64/4 flags for R = 2 while 2 heads
    <= 3 SMs or 5 SMs < 2 heads <= 6 SMs (P4, P5), else 64/8 flags."""
    heads = batch * hv
    r = hv // hk
    if not pro and r == 3 and 3 * heads + batch * hk <= sms < 4 * heads + batch * hk:
        return 96, 4, True
    if 4 * heads <= sms:
        return 32, 4, r <= 4
    if heads <= sms:
        if pro and 2 * heads > sms + 4:
            return 128, 8, False
        if not pro and 2 <= r <= 4 and 104 * sms < 200 * heads <= 113 * sms:
            return 96, 4, True
        return 64, 4, not pro and 3 <= r <= 4 and 3 * heads <= sms
    if pro:
        if r == 2 and (2 * heads <= 3 * sms or 5 * sms < 2 * heads <= 6 * sms):
            return 64, 4, False
        return 64, 8, False
    if r <= 4 and 2 * heads + batch * hk <= 3 * sms:
        return 64, 4, True
    if r == 3 and hk >= 8 and 2 * heads <= 3 * sms:
        return 64, 4, True
    if r <= 4 and (2 * heads <= 3 * sms or 5 * sms < 2 * heads <= 6 * sms):
        return 64, 4, False
    if 20 * heads >= 43 * sms:
        return 128, 8, False
    return 64, 8, False


# SHA256 of (compile key, resolved tune) across token boundaries and N=1..256.
PREFILL_DIGESTS = [
    ((12, 0), 170, 8, 16, "5cd2746a8f4b5171722348a0853e2688e9981e63aa2ba5e5f0b793f338142dd4"),
    ((12, 0), 170, 8, 24, "27505d7820fa999222f3f0f15216e7bc865d2af72ed77dba49e1a319cc80e841"),
    ((12, 0), 170, 16, 32, "2c0b3369679e728c100fb2dc84c0993239c9617f81b8410448c08091d1fc32e0"),
    ((12, 0), 170, 16, 48, "ac4a45a852d97f0449ada0f35926a650b259e37dcb0e2b865f1f124191c86835"),
    ((12, 0), 188, 8, 16, "5cd2746a8f4b5171722348a0853e2688e9981e63aa2ba5e5f0b793f338142dd4"),
    ((12, 0), 188, 8, 24, "27505d7820fa999222f3f0f15216e7bc865d2af72ed77dba49e1a319cc80e841"),
    ((12, 0), 188, 16, 32, "2c0b3369679e728c100fb2dc84c0993239c9617f81b8410448c08091d1fc32e0"),
    ((12, 0), 188, 16, 48, "ac4a45a852d97f0449ada0f35926a650b259e37dcb0e2b865f1f124191c86835"),
    ((12, 1), 48, 8, 16, "7786bc41e6cfc41b5f7f14009449a6750e0baf4ba817e2f483235e873949d6b1"),
    ((12, 1), 48, 8, 24, "2f5ea1c16b71c1ea5c029f63ff06ca22ed2ef78ab12d4378b380e373a1529ee5"),
    ((12, 1), 48, 16, 32, "cd63d0c2ab1b39e6ce6b3fa0e7cc078c3ce2c9bcc6f1356602b2199c0526b248"),
    ((12, 1), 48, 16, 48, "f48d6a5acfe3c5291d36281a434662755ddb62d9a82c11dcd79c7db2dfc37b19"),
]
