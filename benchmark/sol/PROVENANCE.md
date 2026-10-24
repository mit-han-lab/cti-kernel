# Vendored SOL helper and compact ceilings

`sol.py` and `__init__.py` are copies from this project's `kernels/sol-ceilings`
branch (as of `e95f6688`; same copyright, see the SPDX headers), byte-identical except the
SPDX license line, which is MIT here. Change them only on that branch.

`ceilings/{rtx5090,rtx-pro-6000,gb10,gb300}.json` are compact extracts produced by
`compact_ceilings.py`, not byte-identical copies. The full measured tables live
on the `kernels/sol-ceilings` branch at `benchmark/sol/ceilings/`, commit
`e95f6688`: the `7e92a13` read measurements merged with the rotated cold probe
(`probe_transfer.py`, max per size), write/copy from rotated steady-state sources only, and
achieved kernels (`check_achieved.py`); each merged point records its source there.
`gb300` is a partial table (memory and launch only). Each extract's `source` records the
commit and the full-file SHA-256.
The helper and extracts let both GDN SOL models and their CPU tests run without
a sibling checkout.

Regenerate from a directory containing the three full JSON files:

```sh
python benchmark/sol/compact_ceilings.py FULL_DIRECTORY
```

The default output directory is `benchmark/sol/ceilings/`; `--output DIRECTORY`
selects another destination. Input and output directories must differ.

## Hashes

Helper files (verified against `e95f6688`):

| File | SHA-256 |
|---|---|
| `__init__.py` | `ad3195222b50da7013dbd8484a64615994fd1b71c953e4c5c8476eb5f84e1a6d` |
| `sol.py` | `db070f4727aff547ecae5027f44ad9759a23f7ed511f23c189240c456f9e469a` |

Ceiling files (full upstream input and generated compact output):

| File | Full-file SHA-256 | Compact-file SHA-256 | Compact bytes |
|---|---|---|---:|
| `ceilings/rtx5090.json` | `57e0b73337bf3f32dc207cdcf9be1338aaf08fec05233dc5177d054802a51225` | `deffec24009758eb50e5db6a2b739e966795669bfe43c9263763f735393b8e4a` | 8168 |
| `ceilings/rtx-pro-6000.json` | `8e6a3b2525f3e9034268c1137232f4a6c0a335653c560261f00d0c3b65b3c9ef` | `5bbaa6ae7b64c7103e357e2df06f250012ee3e049a76ebd1cff1763fad12dea8` | 8221 |
| `ceilings/gb10.json` | `205d23284b090035fad565ecbf9f1990d718e04ebc90189607be1ca17ad102d1` | `1d156335b06716b407985aec05058da9b0a0375a7c3404f753c27d9e7c75a102` | 8563 |
| `ceilings/gb300.json` | `029995e4bef247191ae8a103adf028d13c528fb9f63d2580082c5ef84a7dc17d` | `acf7f0eaf04693051b036453bc32523608582d07b5863611cb1ecfff623417e5` | 4321 |

## Access-path audit

The projection follows all accesses, including `_required`, `.get()`, validation,
and scheduling helpers, in `sol.py`, `gdn_prefill_sol.py`,
`gdn_prefill_v2_sol.py`, and `test_gdn_prefill{,_v2}_sol.py`:

| Retained path | Consumer |
|---|---|
| `sm_count` | `load_ceilings` validation, `_cta_capacity` residency override, v1 one-wave dispatch, v2 per-SM chain rate and wave tests |
| `l2_bytes` | v1 `build_call` cache budgets, working-set comparisons, and `.get()` diagnostics |
| `missing_measurements` | loader `.get()` and rate-validation exemptions; `estimate` partial-table assumption |
| `memory.{dram_read_gbs,dram_write_gbs,dram_copy_gbs,l2_read_gbs}` | loader validation, `_dram_time`, L2 service time; v2 tests also read copy/L2 rates directly |
| `memory.small_transfer_by_direction.{read,write,copy}[].{bytes,gbs,time_us}` | loader validation; `_dram_time` sorts by bytes and interpolates time in log space |
| `tensor_core.<each of the seven MMA_KINDS>.{tflops_gpu,tflops_per_sm}` | loader validates both rates for every kind; `estimate` uses GPU rate, v2 divides it by SM count |
| `fixed_costs.{graph_node_us,graph_node_pdl_us,eager_launch_us}` | loader validates all three; `estimate` selects the ordinary or PDL floor |
| `scheduling.cta_capacity[].{threads,smem_bytes,regs_per_thread,cta_capacity,theoretical_residency_per_sm,measured_max_concurrent,q_saturation}` | `_cta_capacity` validates and selects shapes; v1 matches with `.get()` and emits these fields plus the original row index |

`device`, `schema_version`, `SPDX-License-Identifier`, and `source` are retained
for identification and provenance. All 19 CTA rows retain their order, and all
points in each read/write/copy curve retain exact numeric values: no rounding
or thinning of interpolated curves. `gbs` and per-SM rates stay because validation
reads them even though interpolation uses `time_us` and compute uses GPU rates.

`memory.small_transfer` is a required, validation-only legacy list, duplicated
from the read curve upstream. It is emitted as an empty list, which the unchanged
loader accepts; all of its computational points remain in the read curve.
Optional `mix_1_1` and `mix_2_1` direction curves are omitted: validation iterates
over present directions, but `Kernel` permits only read/write/copy. No estimate
can select a mixed-direction curve.

Raw measurement metadata and unrelated suites are omitted. `_cta_capacity`
also expands the chosen row with `**chosen` into diagnostic evidence, so direct
helper output loses the raw `measurement` subtree and its echoed assumption.
All numeric estimates and calibrated summary fields remain identical. The v1
model already strips that raw evidence and references the calibrated row;
full paired measurements are available in the upstream tables at that index.
The v1 CLI's `ceiling_sha256` intentionally changes with the file contents.

## CPU equivalence checks

Both full and compact tables pass the two model self-tests (102 v1 device rows;
204 v2 device rows / 612 split variants) and all 230 tests in the two GDN SOL
test modules. Full-versus-compact CLI stdout diffs are empty for both self-tests,
v1 in large/curve modes, v2 JSON ledgers and Markdown for all three devices, and
the shared helper CLI with an explicit residency. The v1 JSON payloads also
compare exactly after removing only `ceiling_sha256`.

Additional exact comparisons cover every retained curve point, interior
interpolation intervals, clamping and sustained-bandwidth boundaries, and all
19 calibrated CTA shapes with all seven MMA kinds and both node floors. Helper
results compare exactly after removing only raw measurement evidence and its
textual echo. The helper source and arithmetic are unchanged.
