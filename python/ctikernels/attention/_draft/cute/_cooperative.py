# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Construction-time cooperative support and conservative compiled-kernel occupancy."""

import torch


def driver():
    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        from cuda import cuda
    return cuda


def checked(result):
    if int(result[0]) != 0:
        raise RuntimeError("CUDA cooperative query failed")
    return result[1]


def supported(device):
    if device.type != "cuda":
        return False
    try:
        cuda = driver()
        index = device.index if device.index is not None else torch.cuda.current_device()
        dev = checked(cuda.cuDeviceGet(index))
        return bool(
            checked(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, dev
                )
            )
        )
    except (ImportError, AttributeError, RuntimeError):
        return False


def blocks_per_sm(compiled, device, threads, shared_bytes):
    """CuTe 4.8 library -> CUkernel -> CUfunction; unknown APIs use one CTA/SM.

    shared_bytes includes the allocator's 1 KiB reserve. Subtract actual static
    storage to obtain a conservative dynamic allocation for the occupancy query.
    The cooperative driver launch remains the final resource-fit check.
    """
    try:
        cuda = driver()
        with torch.cuda.device(device):
            library = cuda.CUlibrary(int(compiled.library))
            if checked(cuda.cuLibraryGetKernelCount(library)) != 1:
                return 1
            kernel = checked(cuda.cuLibraryEnumerateKernels(1, library))[0]
            function = checked(cuda.cuKernelGetFunction(kernel))
            static = checked(
                cuda.cuFuncGetAttribute(
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function
                )
            )
            return checked(
                cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    function, threads, max(0, shared_bytes - static)
                )
            )
    except Exception:  # noqa: BLE001 -- optional executor/driver introspection
        # CuTe DSLRuntimeError derives directly from Exception. In particular,
        # TVM FFI may expose library without implementing its lazy-load path.
        return 1


def launch_option_error(error):
    """Only launch-option failures qualify for a different compiled variant."""
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "cooperative",
            "programmatic",
            "use_pdl",
            "cuda_error_not_supported",
            "cuda_error_invalid_value",
            "cudaerrornotsupported",
            "cudaerrorinvalidvalue",
        )
    )
