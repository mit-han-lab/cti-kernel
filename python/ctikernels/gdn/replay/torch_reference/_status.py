# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU semantic implementation of the device lifecycle."""

import torch


def lifecycle(status, mode, expected=0):
    seq = int(status.sequence[0])
    views = [
        (r[:8].view(torch.int64), r[8:12].view(torch.int32), r[12:].view(torch.int32))
        for r in status.records
    ]
    good = seq > 0
    for s, c, d in views:
        if mode == "begin":
            good &= int(s[0]) == int(c[0]) == int(d[0]) == 0
        elif mode == "clear":
            good &= int(s[0]) == expected == seq and int(d[0]) == 1
        else:
            good &= int(s[0]) == seq
    if not good:
        status.poison[0] |= 64
        return
    for s, c, d in views:
        if mode == "begin":
            s[0], c[0], d[0] = seq, 32 if int(status.poison[0]) else 0, 0
        elif mode == "end":
            if int(status.poison[0]):
                c[0] |= 32
            d[0] = 1
        else:
            s.zero_()
            c.zero_()
            d.zero_()
    if mode in ("begin", "clear"):
        status.validity.zero_()


def guard(status):
    seq = int(status.sequence[0])
    if (
        seq <= 0
        or int(status.record[:8].view(torch.int64)[0]) != seq
        or int(status.record[12:].view(torch.int32)[0]) != 0
    ):
        status.poison[0] |= 64
        return False
    if int(status.poison[0]):
        status.code[0] |= 32
        return False
    return True
