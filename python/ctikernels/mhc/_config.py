# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Compile-time environment controls shared by the host cache and CuTe kernels."""

import os


def kernel_options():
    chain = os.environ.get("CTI_MHC_CHAIN", "lane16")
    exchange = os.environ.get("CTI_MHC_EXCHANGE", "last")
    fence = os.environ.get("CTI_MHC_FENCE", "single")
    proj = os.environ.get("CTI_MHC_PROJ", "split")
    if fence not in ("single", "all"):
        raise ValueError("CTI_MHC_FENCE must be single or all")
    if proj not in ("split", "token"):
        raise ValueError("CTI_MHC_PROJ must be split or token")
    publish = os.environ.get("CTI_MHC_PUBLISH", "early")
    if publish not in ("early", "late"):
        raise ValueError("CTI_MHC_PUBLISH must be early or late")
    if chain not in ("serial", "lane16"):
        raise ValueError("CTI_MHC_CHAIN must be serial or lane16")
    if exchange not in ("last", "all"):
        raise ValueError("CTI_MHC_EXCHANGE must be last or all")
    # Decode trigger placement: 1 = after the PDL wait, residual = after the
    # first piece's residual phase, exchange = before the arrival, 0 = at the end.
    # Other kernels treat every value except 0 as the after-wait trigger.
    trigger = os.environ.get("CTI_MHC_EARLY_TRIGGER", "1")
    if trigger not in ("1", "0", "residual", "exchange"):
        raise ValueError("CTI_MHC_EARLY_TRIGGER must be 1, 0, residual or exchange")
    prefetch = os.environ.get("CTI_MHC_AR_PREFETCH", "1")
    if prefetch not in ("1", "0"):
        raise ValueError("CTI_MHC_AR_PREFETCH must be 1 or 0")
    return (
        os.environ.get("CTI_MHC_IKET", "0") == "1",
        trigger,
        os.environ.get("CTI_MHC_FASTDIV", "0") == "1",
        chain,
        exchange,
        publish,
        fence,
        proj,
        prefetch == "1",
    )
