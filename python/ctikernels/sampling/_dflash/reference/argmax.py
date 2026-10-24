# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Torch-only CPU reference for RowArgmax."""

import torch


class Provider:
    def __init__(self, device):
        self.device = device

    def warm_up(self, logits, out, proposals=None, block_size=2):
        pass

    def run(self, logits, out, workspace, proposals=None, block_size=None):
        # Chain mode: every row holds its first argmax, one allowed value.
        if out.dtype == torch.int64:
            torch.argmax(logits, dim=-1, out=out)
        else:
            # Torch's out=argmax only accepts int64; reference-only temporary.
            out.copy_(torch.argmax(logits, dim=-1))
