# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Inference sampling operations."""

from ._dflash.argmax import RowArgmax
from ._dflash.ordinary import ORDINARY, RequestLocalSampling
from ._logprobs import TopLogprobs
from ._penalties import LogitPenalties
from .inference import (
    CategoricalSampling,
    ChainSampling,
    RequestLocalCategorical,
    RequestLocalChainSampling,
    steer_logits_,
    topk_by_id,
)

__all__ = [
    "steer_logits_",
    "topk_by_id",
    "TopLogprobs",
    "LogitPenalties",
    "RequestLocalSampling",
    "ORDINARY",
    "RowArgmax",
    "CategoricalSampling",
    "ChainSampling",
    "RequestLocalCategorical",
    "RequestLocalChainSampling",
]
