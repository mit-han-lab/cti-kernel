# kernels/draft-window-rc validation (2026-10-16)

Branch `kernels/draft-window-rc` = `kernels/rc-0.1.13` (DW1 core d6fe743 already included) plus the rc3 delta
(c1a5403..8c486be cherry-picked without conflicts), the long-context rows and the SGLang range table.

GPU, default selection (`-m "not exhaustive"`), shipped tree at this branch:

| suite | RTX 5090 | RTX PRO 6000 | GB10 |
|---|---|---|---|
| draft core (attention, capture_trace, cute, inline, integration) + test_dflash_draft_window.py + chunked-reference audit | 452 passed | 452 passed | 452 passed |
| test_dflash_long_context.py (C=131,072 / 262,144, B=1-4, CuTe + FA2 DRAFT for the four draft masks, FixedQuery verify hd256/hd128, eager + two graph replays, chunked FP32 reference, B*C < 2^31 asserted) | 80 passed | 80 passed | 80 passed |

No failures and no skips. The long-context file runs in 40-45 s per device (256 MiB physical K/V pool per case;
the logical token and page CSR spans the full B*C).

Performance: unchanged from the rc3 receipts (`draft_window_dw1_rc3.md`). The 12 windowed
pair rows at 0.992-0.998 vs 0.1.12 were accepted.
