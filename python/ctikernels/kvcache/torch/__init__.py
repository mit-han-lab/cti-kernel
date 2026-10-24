# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU reference for kvcache row copies."""


class Provider:
    name = "torch"

    def __init__(self, tensors):
        self.tensors = tensors

    def warm_up(self, index_dtypes):
        pass

    def __call__(self, sources, targets):
        for t in self.tensors:
            rows = t.shape[0]
            keep = (sources >= 0) & (targets >= 0) & (sources < rows) & (targets < rows)
            t.index_copy_(0, targets[keep].long(), t.index_select(0, sources[keep].long()))
