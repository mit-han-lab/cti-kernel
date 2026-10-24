# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import torch


class ChainSampling:
    """Linear speculative sampling using the actual filtered proposal distribution.

    Probability math is FP32. Inputs are immutable; counts and RNG offsets are
    caller-owned outputs. Single-element CUDA seed/offset tensors identify a
    shared Philox stream (batch rows are subsequences), NOT per-request seeds.
    Calls execute on the current stream and can be captured in CUDA graphs.
    Empty batches return empty outputs without a launch or an offset advance.
    """

    def __init__(self) -> None:
        import flashinfer.sampling as provider

        self._provider = provider

    @staticmethod
    def _require_rng(probabilities: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor) -> None:
        if (
            not probabilities.is_cuda
            or probabilities.dtype != torch.float32
            or not probabilities.is_contiguous()
            or probabilities.requires_grad
        ):
            raise ValueError("probabilities must be contiguous, non-differentiable CUDA FP32")
        for tensor in (seed, offset):
            if (
                tensor.device != probabilities.device
                or tensor.dtype != torch.int64
                or tensor.shape != (1,)
                or not tensor.is_contiguous()
            ):
                raise ValueError("seed and offset must be single-element CUDA int64 tensors")

    def probabilities(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize top-k-then-top-p probabilities, as in ordinary sampling.

        The caller supplies positive temperatures, k in [1, vocabulary] and
        p in (0, 1]. Unlike a fused draw, verification needs the complete q/p.
        """
        if logits.shape[0] == 0:
            return torch.empty_like(logits, dtype=torch.float32)
        probabilities = self._provider.softmax(logits.float(), temperatures, enable_pdl=False)
        probabilities = self._provider.top_k_renorm_probs(probabilities, top_k)
        return self._provider.top_p_renorm_probs(probabilities, top_p)

    def sample(
        self, probabilities: torch.Tensor, *, seed: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        self._require_rng(probabilities, seed, offset)
        if probabilities.ndim != 2:
            raise ValueError("categorical probabilities must have shape [tokens, vocabulary]")
        if probabilities.shape[0] == 0:
            return probabilities.new_empty(0, dtype=torch.int32)
        tokens = self._provider.sampling_from_probs(probabilities, seed=seed, offset=offset)
        # The unfiltered categorical provider consumes ONE uniform per row.
        # Reserve a whole Philox4 block, independent of vocabulary/scan length.
        offset.add_(4)
        return tokens

    def verify(
        self,
        draft_probabilities: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_probabilities: torch.Tensor,
        *,
        accepted_counts: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B,Q] valid prefix+bonus, followed by -1; counts EXCLUDE bonus.

        q is [B,Q-1,V], p [B,Q,V], proposals [B,Q-1]. Distributions must
        be normalized and proposals drawn from q. Counts reset on every call;
        they describe the emitted prefix, not hypothetical later acceptances.
        Q includes the anchor and must be >= 1; there is no wrapper upper cap.
        """
        self._require_rng(draft_probabilities, seed, offset)
        self._require_rng(target_probabilities, seed, offset)
        if target_probabilities.ndim != 3:
            raise ValueError("target probabilities must have shape [B,Q,V]")
        batch_size, block_size, vocabulary = target_probabilities.shape
        if (
            block_size < 1
            or vocabulary < 1
            or draft_probabilities.shape != (batch_size, block_size - 1, vocabulary)
            or draft_token_ids.shape != (batch_size, block_size - 1)
            or accepted_counts.shape != (batch_size,)
        ):
            raise ValueError("chain sampling requires matching q/p/proposals/counts and Q >= 1")
        for tensor in (draft_token_ids, accepted_counts):
            if (
                tensor.device != seed.device
                or tensor.dtype != torch.int32
                or not tensor.is_contiguous()
            ):
                raise ValueError("proposals and counts must be contiguous CUDA int32")
        if batch_size == 0:
            return draft_token_ids.new_empty((0, block_size))
        # The provider launches one CTA per batch row, including zero drafts.
        # Its empty acceptance loops then draw the bonus with the original RNG/CDF.
        accepted_counts.zero_()
        tokens, _, _ = self._provider.chain_speculative_sampling(
            draft_probabilities,
            draft_token_ids,
            target_probabilities,
            maybe_output_emitted_draft_token_num=accepted_counts,
            deterministic=True,
            seed=seed,
            offset=offset,
        )
        # Provider consumes <= Q+1 uniforms: Q-1 candidate tests, potentially
        # one repeated rejection test for its diagnostic count, then the bonus.
        # Reserve whole Philox4 blocks. No overlap across changing B or Q.
        offset.add_(((block_size + 4) // 4) * 4)
        return tokens
