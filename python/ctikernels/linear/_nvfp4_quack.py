# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""QuACK's W4A4 (NVFP4 x NVFP4) GEMM, used through ``quack-kernels``' public epilogue API
(Apache-2.0; https://github.com/Dao-AILab/quack). No QuACK source is copied: the two
epilogues below are composed with its ``gemm_epilogue`` frontend and run by its SM120
block-scaled kernel (``mma.sync`` ``kind::mxf4nvf4``), which also serves SM121.

- ``linear``: ``D = acc * column_scale[n] * input_scale`` rounded once to the output dtype.
- ``gated``: on a weight whose rows interleave gate and up (``g0, u0, g1, u1, ...``), with
  ``s[n] = column_scale[n] * input_scale``,
  ``silu(acc_g * s[2j]) * (acc_u * s[2j + 1])`` in FP32, quantized to
  NVFP4 in the epilogue (QuACK's block-scale-factor store: E4M3 scale per 16 outputs, E2M1
  values, ``norm_const`` = 1 / the output's global scale folded in).

``load(device)`` returns this module's launcher for a device type it is qualified on, or None:
another device type, ``CTI_NVFP4_QUACK=0``, or quack-kernels / nvidia-cutlass-dsl not at the
pinned versions. Only a resolution that can use QuACK imports it.
"""

from __future__ import annotations

import importlib.metadata
import os
import threading
import warnings
from typing import NamedTuple

import torch

VARIABLE = "CTI_NVFP4_QUACK"
DISTRIBUTIONS = {"quack-kernels": "0.6.5", "nvidia-cutlass-dsl": "4.8.0"}
# The device types the kernels are qualified on: (name, capability).
DEVICES = frozenset(
    {
        ("NVIDIA GeForce RTX 5090", (12, 0)),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0)),
        ("NVIDIA GB10", (12, 1)),
    }
)


class Config(NamedTuple):
    """A QuACK SM120 block-scaled tile configuration (one CTA per cluster)."""

    tile_m: int
    tile_n: int
    pingpong: bool
    dynamic: bool  # dynamic persistent (cluster launch control) or static persistent


DEFAULT = Config(128, 128, True, True)


class Launcher:
    """Launches the two epilogues; imports QuACK on construction."""

    def __init__(self) -> None:
        from . import _nvfp4_quack_epilogues as epilogues

        self._epilogues = epilogues

    def linear(
        self,
        a_data: torch.Tensor,
        a_scale: torch.Tensor,
        b_data: torch.Tensor,
        b_scale: torch.Tensor,
        column_scale: torch.Tensor,
        input_scale: torch.Tensor,
        out: torch.Tensor,
        config: Config,
    ) -> None:
        """``out[m, n] = (A . B^T)[m, n] * column_scale[n] * input_scale``; operands as
        ``gated``."""
        self._run(
            self._epilogues.linear,
            a_data,
            a_scale,
            b_data,
            b_scale,
            out.unsqueeze(0),
            {"column_scale": column_scale.unsqueeze(0), "input_scale": input_scale},
            config,
        )

    def gated(
        self,
        a_data: torch.Tensor,
        a_scale: torch.Tensor,
        b_data: torch.Tensor,
        b_scale: torch.Tensor,
        column_scale: torch.Tensor,
        input_scale: torch.Tensor,
        norm_const: torch.Tensor,
        out_data: torch.Tensor,
        out_scale: torch.Tensor,
        config: Config,
    ) -> None:
        """A ``[M, K / 2]`` and B ``[N, K / 2]`` packed E2M1 (``float4_e2m1fn_x2``) with blocked
        E4M3 scales; ``column_scale`` FP32 ``[N]``; ``input_scale`` and ``norm_const`` FP32 ``[1]``; ``out_data``
        ``[M, N / 4]`` packed E2M1 and ``out_scale`` blocked E4M3 for ``[M, N / 2]``."""
        self._run(
            self._epilogues.gated,
            a_data,
            a_scale,
            b_data,
            b_scale,
            None,
            {
                "column_scale": column_scale.unsqueeze(0),
                "input_scale": input_scale,
                "postact": out_data.unsqueeze(0),
                "postact_sf": out_scale.unsqueeze(0),
                "sfd_norm_const": norm_const,
            },
            config,
        )

    @staticmethod
    def _run(epilogue, a_data, a_scale, b_data, b_scale, out, arguments, config) -> None:
        with torch.cuda.device(a_data.device):
            epilogue.gemm(
                a_data.unsqueeze(0),
                b_data.unsqueeze(0),
                out,
                epi_args=arguments,
                tile_M=config.tile_m,
                tile_N=config.tile_n,
                cluster_M=1,
                cluster_N=1,
                pingpong=config.pingpong,
                persistent=True,
                is_dynamic_persistent=config.dynamic,
                SFA=a_scale.unsqueeze(0),
                SFB=b_scale.unsqueeze(0),
                bs_format_a="nvfp4",
                bs_format_b="nvfp4",
                post_init_attrs=(("use_pdl", True),),
            )


_launchers: dict[torch.device, Launcher | None] = {}
_lock = threading.Lock()


def load(device: torch.device) -> Launcher | None:
    with _lock:
        if device not in _launchers:
            _launchers[device] = _resolve(device)
        return _launchers[device]


def _resolve(device: torch.device) -> Launcher | None:
    if device.type != "cuda" or os.environ.get(VARIABLE, "1") == "0":
        return None
    name = torch.cuda.get_device_name(device)
    if (name, torch.cuda.get_device_capability(device)) not in DEVICES:
        return None
    for distribution, version in DISTRIBUTIONS.items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None
        if installed != version:
            warnings.warn(
                f"{distribution} {installed} is installed, not {version}: NVFP4 GEMMs on "
                f"{name} use the dequantizing fallback",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
    return Launcher()
