// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
// Adapted from FlashMLA (sgl-project fork) 05e26647fe840b8baedae486c2d86d5ce4efeb7c
// (`csrc/kerutils/include/kerutils/device/device.cuh`), unmodified;
// DSV4 stopgap, replace later; see THIRD_PARTY_NOTICES.

#pragma once

#include "kerutils/common/common.h"

#include "common.h"
#include "sm80/intrinsics.cuh"
#include "sm80/helpers.cuh"
#include "sm90/intrinsics.cuh"
#include "sm90/helpers.cuh"
#include "sm100/intrinsics.cuh"
#include "sm100/helpers.cuh"
#include "sm100/gemm.cuh"
#include "sm100/tma_cta_group2_nosplit.cuh"
