// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
// Adapted from FlashMLA (sgl-project fork) 05e26647fe840b8baedae486c2d86d5ce4efeb7c
// (`csrc/sm100/prefill/sparse/fwd/head64/phase1.h`), unmodified;
// DSV4 stopgap, replace later; see THIRD_PARTY_NOTICES.

#pragma once

#include "params.h"

namespace sm100::fwd::head64 {

template<int D_QK>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params);

}
