# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/engine/sample.py`, sampling routing); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import torch


class CategoricalSampling:
    """Temperature softmax and categorical draws over a batch of probability rows.

    The provider module is resolved once at construction and never exposed to
    callers; individual functions are looked up on the call path that needs
    them, so an incomplete provider fails where it is used, exactly as before
    this class existed. Top-k and top-p filters are applied by the provider's
    fused samplers, so no filtered probability tensor is materialized. Empty
    batches return empty outputs without a launch.
    """

    def __init__(self) -> None:
        import flashinfer.sampling as provider

        self._provider = provider

    def softmax(
        self, logits: torch.Tensor, temperatures: torch.Tensor, *, enable_pdl: bool
    ) -> torch.Tensor:
        if logits.shape[0] == 0:
            return torch.empty_like(logits, dtype=torch.float32)
        return self._provider.softmax(logits, temperatures, enable_pdl=enable_pdl)

    def sample(
        self,
        probabilities: torch.Tensor,
        *,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if probabilities.shape[0] == 0:
            return probabilities.new_empty(0, dtype=torch.int32)
        if top_k is None and top_p is None:
            return self._provider.sampling_from_probs(probabilities)
        if top_p is None:
            return self._provider.top_k_sampling_from_probs(probabilities, top_k)
        if top_k is None:
            return self._provider.top_p_sampling_from_probs(probabilities, top_p)
        return self._provider.top_k_top_p_sampling_from_probs(probabilities, top_k, top_p)
