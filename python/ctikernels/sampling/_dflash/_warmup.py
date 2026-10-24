# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Validated warm-up requests and storage-free tensor descriptions."""

import torch

DTYPES = (torch.bfloat16, torch.float16, torch.float32)
FILTERS = ("none", "top_k", "top_p", "both", "auto")
MODES = ("point", "dense", "sparse", "sample", "candidates", "cursors")


def capacity(value):
    if not isinstance(value, (int, bool)):
        try:
            values = tuple(value)
        except TypeError as error:
            raise ValueError("invalid batch capacity") from error
        if not values:
            raise ValueError("batch sizes must not be empty")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 2**31
            for item in values
        ):
            raise ValueError("invalid batch capacity")
        value = max(values)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**31:
        raise ValueError("invalid batch capacity")
    return value


def choices(values, allowed, name):
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError(f"invalid {name}") from error
    if any(v not in allowed or (name == "min_p" and not isinstance(v, bool)) for v in values):
        raise ValueError(f"invalid {name}")
    return tuple(dict.fromkeys(values))


def tensor(dtype, *shape):
    return torch.empty(shape, dtype=dtype, device="meta")


def logits_layouts(dtype, *shape, strided=True):
    """Dense logits, plus (strided=True) a row-strided view selecting that variant."""
    yield tensor(dtype, *shape)
    if strided:
        yield tensor(dtype, *shape[:-1], shape[-1] + 1)[..., : shape[-1]]


def probability_variants(op, batch_capacity, dtypes, filters, min_p, strided=True):
    # identity_group (and the compact reservation, rows <= SMs) are monotone in
    # B*L, so these endpoints cover their ranges. Two rows reach the strided
    # variant with the reservation. L > 65535 (flat grid) only occurs with group 2
    # and no reservation; keep the widest grid.y L too.
    points = (
        (1, 1),
        (min(batch_capacity, 2), 1),
        (1, min(op.block_size, 2)),
        (batch_capacity, min(op.block_size, 65535)),
        (batch_capacity, op.block_size),
    )
    for b, length in dict.fromkeys(points):
        temps = tensor(torch.float32, b)
        slots = tensor(torch.int32, b)
        work = tensor(torch.uint8, 8)
        out = tensor(torch.float32, b, length, op.vocab_size)
        for dtype in dtypes:
            for logits in logits_layouts(dtype, b, length, op.vocab_size, strided=strided):
                for hint in filters:
                    for use in min_p:
                        yield (
                            "probabilities_min_p" if use else "probabilities",
                            (logits, temps, slots, temps, slots, out, work),
                            hint,
                        )
        if True in min_p and dtypes and filters:
            yield "min_p_renormalize", (out, temps, temps, slots, work), 0


def chain_variants(op, modes):
    b, q, v = 1, op.block_size, op.vocab_size
    p = tensor(torch.float32, b, q, v)
    drafts = tensor(torch.int32, b, q - 1)
    dense = tensor(torch.float32, b, q - 1, v)
    sparse = tensor(torch.float32, b, q - 1, 16)
    ids = tensor(torch.int32, b, q - 1, 16)
    temps, slots = tensor(torch.float32, b), tensor(torch.int32, b)
    seeds, cursors = tensor(torch.uint32, b, 2), tensor(torch.int64, b)
    work, out = tensor(torch.uint8, 8), tensor(torch.int32, b, q)
    counts = tensor(torch.int32, b)
    common = (temps, slots, seeds, cursors, work, out, counts)
    for mode in modes:
        if mode == "point":
            yield mode, (p, drafts, *common), 0
        elif mode == "dense":
            yield mode, (p, drafts, dense, *common), 0
        elif mode == "sparse":
            yield mode, (p, drafts, sparse, ids, *common), 0
        elif mode == "sample" and q > 1:
            yield mode, (dense, temps, slots, seeds, cursors, work, drafts), 0
        elif mode == "candidates" and q > 1 and v >= 16:
            scores = tensor(torch.float32, b, 16)
            candidates = tensor(torch.int32, b, 16)
            for position in (*range(q - 1), None):
                yield (
                    mode,
                    (
                        scores,
                        candidates,
                        temps,
                        slots,
                        seeds,
                        cursors,
                        work,
                        counts,
                        scores,
                        tensor(torch.int32) if position is None else slots,
                    ),
                    position,
                )
        elif mode == "cursors":
            yield mode, (cursors, cursors, slots, cursors), 0


def ordinary_batches(vocab, batch_capacity, sms):
    """Batch sizes reaching every (group, looped) configuration up to capacity.

    identity_group flips to 2 at B*chunks >= 4*SMs; the P4/D1 loop exceeds one
    group at B*groups >= 16*SMs; the compact reservation needs B <= SMs. All are
    monotone in B; include each threshold
    and its predecessor (extra points only deduplicate to the same keys).
    """
    chunks = (vocab + 4095) // 4096
    # The compact CTA reservation stops above one row per SM.
    points = {1, batch_capacity, sms, sms + 1}
    for target, width in ((4 * sms, chunks), (16 * sms, chunks), (16 * sms, (chunks + 1) // 2)):
        first = -(-target // max(1, width))
        points.update((first - 1, first))
    return sorted(b for b in points if 1 <= b <= batch_capacity)


def ordinary_variants(op, batch_capacity, dtypes, filters, all_greedy, sms=None, strided=True):
    batches = (
        (1, batch_capacity) if sms is None else ordinary_batches(op.vocab_size, batch_capacity, sms)
    )
    for b in dict.fromkeys(batches):
        temps, slots = tensor(torch.float32, b), tensor(torch.int32, b)
        seeds, positions = tensor(torch.uint32, b, 2), tensor(torch.int64, b)
        work, out = tensor(torch.uint8, 8), tensor(torch.int32, b)
        for dtype in dtypes:
            for logits in logits_layouts(dtype, b, op.vocab_size, strided=strided):
                for hint in filters:
                    for use in (False, True):
                        minimum = temps if use else None
                        if hint == "none":
                            yield (
                                "ordinary_identity",
                                (
                                    logits,
                                    temps,
                                    minimum,
                                    slots,
                                    seeds,
                                    positions,
                                    work,
                                    work,
                                    out,
                                ),
                                0,
                            )
                        else:
                            yield (
                                "ordinary_filtered",
                                (
                                    logits,
                                    temps,
                                    slots,
                                    temps,
                                    minimum,
                                    slots,
                                    seeds,
                                    positions,
                                    tensor(torch.float32, b, op.vocab_size),
                                    work,
                                    work,
                                    out,
                                ),
                                hint,
                            )
                if all_greedy:
                    yield (
                        "ordinary_identity",
                        (
                            logits,
                            temps,
                            None,
                            slots,
                            seeds,
                            positions,
                            work,
                            work,
                            out,
                        ),
                        "greedy",
                    )
