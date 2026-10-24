// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
// Adapted from FlashMLA (sgl-project fork) 05e26647fe840b8baedae486c2d86d5ce4efeb7c
// (`csrc/kerutils/include/kerutils/common/common.h`), unmodified;
// DSV4 stopgap, replace later; see THIRD_PARTY_NOTICES.

#pragma once

namespace kerutils {}

#define KU_PRINTLN(fmt, ...) { cute::print(fmt, ##__VA_ARGS__); print("\n"); }

namespace ku = kerutils;

