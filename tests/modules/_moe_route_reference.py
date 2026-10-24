# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Torch reference of DeepSeek-V4 routing (DeepSeek-V4-Flash 60d8d707 inference/model.py
Gate; SGLang 0.5.22 srt/layers/moe/topk.py biased_topk_impl and hash_topk.py HashTopK)."""

import torch
import torch.nn.functional as F


def scores(x, router_w):
    """sqrt(softplus(.)) of the FP32 logits."""
    return F.softplus(x.float() @ router_w.float().T).sqrt()


def _weights(s, ids, renormalize, scale):
    w = s.gather(1, ids.long())
    if renormalize:
        w = w / (w.sum(dim=1, keepdim=True) + 1e-20)
    return w * scale


def dsv4_route(x, router_w, top_k, bias=None, *, renormalize=True, scale=1.0, valid=None):
    s = scores(x, router_w)
    key = s + bias if bias is not None else s
    ids = torch.sort(key, dim=1, descending=True, stable=True).indices[:, :top_k]
    w = _weights(s, ids, renormalize, scale)
    return _mask(w, ids.int(), valid)


def dsv4_hash_route(x, router_w, token_ids, table, *, renormalize=True, scale=1.0, valid=None):
    ids = table[token_ids.long()]
    w = _weights(scores(x, router_w), ids, renormalize, scale)
    return _mask(w, ids.int(), valid)


def _mask(w, ids, valid):
    if valid is None:
        return w, ids
    live = torch.arange(w.shape[0], device=w.device)[:, None] < int(valid)
    return w.where(live, 0.0), ids.where(live, -1)


def bias_margin(x, router_w, top_k, bias):
    """Per row, the gap between the k-th and (k+1)-th selection keys (near ties can flip)."""
    s = scores(x, router_w)
    key = (s + bias if bias is not None else s).sort(dim=1, descending=True).values
    return key[:, top_k - 1] - key[:, top_k]
