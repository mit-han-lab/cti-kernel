# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU checks of offline compiler matrix coverage, without importing the DSL."""

import runpy
from pathlib import Path

import pytest

from ctikernels.gdn import _cute_prefill


@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
@pytest.mark.parametrize("hv", [32, 48])
def test_output_matrix_covers_precision_pairs_and_geometry_defaults(cc, hv):
    compiler = runpy.run_path(
        Path(__file__).resolve().parents[2] / "scripts/compile_gdn_cute_prefill.py"
    )
    configs = [
        _cute_prefill.read_tune(value, cc, (8 if hv in (16, 24) else 16, hv))
        for value in compiler["output_matrix_tunes"]([""])
    ]
    assert len(configs) == 12
    legacy = [c for c in configs if c["o_form"] == "legacy"]
    assert len(legacy) == 8
    assert {(c["overlap"], c["aq_qk"], c["aq_out"]) for c in legacy} == {
        (overlap, qk, out) for overlap in (0, 1) for qk in (0, 1) for out in (0, 1)
    }
    for overlap in (0, 1):
        assert (
            _cute_prefill.read_tune(f"overlap={overlap}", cc, (8 if hv in (16, 24) else 16, hv))
            in legacy
        )
    pre = [c for c in configs if c["o_form"] == "pre"]
    assert len(pre) == 4
    assert {(c["overlap"], c["qp_split"]) for c in pre} == {
        (overlap, split) for overlap in (0, 1) for split in (0, 1)
    }
