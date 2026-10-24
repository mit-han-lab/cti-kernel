# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Lazy CPU extension and store-only bootstrap (no CUDA imports)."""

import functools
import os
import time
from datetime import timedelta
from pathlib import Path

from ctikernels.utils import load_aot

# Where rdma-core installs its headers; CTI_VERBS_INCLUDE overrides.
_VERBS_INCLUDE_DIRS = ("/usr/include", "/usr/local/include")


def verbs_include_flags(candidates=None) -> list[str]:
    """``-idirafter <dir>`` for the first directory holding infiniband/verbs.h.

    A toolchain with its own sysroot (conda compilers) does not search the system include
    directory; searching it last adds the verbs headers without shadowing the sysroot's.
    Header types only: the library itself is opened at runtime.
    """
    if candidates is None:
        override = os.environ.get("CTI_VERBS_INCLUDE")
        candidates = (override,) if override else _VERBS_INCLUDE_DIRS
    for directory in candidates:
        if (Path(directory) / "infiniband" / "verbs.h").is_file():
            return ["-idirafter", str(directory)]
    return []


@functools.cache
def module():
    return load_aot(
        "two_node_proxy",
        cpp_files=["two_node_proxy.cpp"],
        extra_cflags=verbs_include_flags(),
        extra_ldflags=["-ldl"],
    )


def available() -> str:
    try:
        return str(module().two_node_proxy_available())
    except Exception as error:  # noqa: BLE001 - capability probing must not raise
        return f"two-node proxy extension unavailable ({type(error).__name__})"


@functools.cache
def _proxy_class():
    import tvm_ffi

    module()

    @tvm_ffi.register_object("ctiengine.TwoNodeProxy")
    class Proxy(tvm_ffi.Object):
        def __init__(self):
            self.__ffi_init__()

    return Proxy


def create(ibdev, gid_index, slot_bytes):
    proxy = _proxy_class()()
    proxy.init(ibdev, gid_index, slot_bytes)
    return proxy


def barrier(store, key, timeout_s=60):
    """Single-use two-rank barrier; callers must supply a fresh key."""
    store.add(key, 1)
    deadline = time.monotonic() + timeout_s
    while int(store.get(key)) < 2:
        if time.monotonic() >= deadline:
            raise TimeoutError("two-node store barrier timed out")
        time.sleep(0.001)


def agree(store, key, rank, reason, timeout_s=60):
    """Two-rank provider agreement under a fresh key.

    Each rank publishes "" (usable) or its reason. Returns "" when both ranks
    are usable, else the failure reasons tagged by rank, identically on both.
    A peer that never answers raises TimeoutError after timeout_s.
    """
    peer = 1 - rank
    store.set(f"{key}/{rank}", ("fail:" + reason if reason else "ok").encode())
    try:
        store.wait([f"{key}/{peer}"], timedelta(seconds=timeout_s))
        answer = bytes(store.get(f"{key}/{peer}")).decode()
    except Exception as error:
        raise TimeoutError(
            f"two-node peer rank {peer} did not answer {key} within {timeout_s} s"
            f" ({type(error).__name__})"
        ) from error
    reasons = {rank: reason, peer: answer[5:] if answer.startswith("fail:") else ""}
    return "; ".join(f"rank {r}: {reasons[r]}" for r in (0, 1) if reasons[r])


def bootstrap(proxy, store, prefix, rank, slot_bytes, timeout_s=60):
    """Validate equal layouts, exchange opaque network-order endpoints, connect.

    The caller's agreement after connecting replaces a connected barrier.
    """
    peer = 1 - rank
    config = f"b1-v1:{slot_bytes}".encode()
    store.set(f"{prefix}/{rank}/config", config)
    store.set(f"{prefix}/{rank}/ep", bytes(proxy.local_endpoint()))
    keys = [f"{prefix}/{peer}/config", f"{prefix}/{peer}/ep"]
    store.wait(keys, timedelta(seconds=timeout_s))
    if store.get(keys[0]) != config:
        raise ValueError("two-node ranks must use the same max_bytes and protocol")
    proxy.connect(store.get(keys[1]))
