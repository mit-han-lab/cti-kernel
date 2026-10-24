# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Isolated FI 0.7 norm overlay; preserve the runner's Torch/CUDA/CuTe stack."""

import importlib.metadata
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# Pure-Python requirements in the 0.7.0.post1 METADATA, plus requests' dependencies.
# CUDA tile, NCCL and cuDNN are optional to the norm import; never replace the
# runner's native stack to satisfy unrelated FlashInfer features.
PURE_PYTHON = {
    "click",
    "einops",
    "nvidia-ml-py",
    "packaging",
    "requests",
    "tabulate",
    "tqdm",
    "certifi",
    "charset-normalizer",
    "idna",
    "urllib3",
}


def missing_pure_requirements(requirements):
    missing = []
    for text in requirements:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        if canonicalize_name(requirement.name) not in PURE_PYTHON:
            continue
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version is None or not requirement.specifier.contains(version):
            missing.append(str(requirement))
    return missing


def install_flashinfer(target, version):
    """Install with --no-deps and prove the child imports the requested overlay."""
    pip = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--disable-pip-version-check",
        "--target",
        str(target),
    ]
    subprocess.run(
        pip + [f"flashinfer-python=={version}"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    distributions = list(importlib.metadata.distributions(path=[str(target)]))
    metadata = next(d for d in distributions if d.metadata["Name"] == "flashinfer-python")
    requirements = list(metadata.requires or [])
    # Requests is an existing FI requirement, but a minimal runner may lack its
    # small Python dependencies too. Read installed metadata where available.
    try:
        requirements += importlib.metadata.requires("requests") or []
    except importlib.metadata.PackageNotFoundError:
        requirements += [
            "certifi>=2017.4.17",
            "charset-normalizer<4,>=2",
            "idna<4,>=2.5",
            "urllib3<3,>=1.21.1",
        ]
    missing = missing_pure_requirements(requirements)
    if missing:
        subprocess.run(
            pip + missing, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; import importlib.metadata as m; "
            "sys.path.insert(0,sys.argv[1]); import flashinfer.norm as norm; "
            "assert m.version('flashinfer-python') == sys.argv[2]; "
            "assert Path(norm.__file__).is_relative_to(Path(sys.argv[1])); "
            "assert callable(norm.fused_add_rmsnorm); "
            "assert callable(norm.gemma_fused_add_rmsnorm); "
            "print('FI', sys.argv[2], 'isolated norm import: OK', flush=True)",
            str(target),
            version,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def install_fi07(target):
    """Compatibility entry point for our provider qualification overlay."""
    return install_flashinfer(target, "0.7.0.post1")
