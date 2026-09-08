# compstate — portable P0/P2 diagnostics

Minimal, path-parameterised copy of the composition/state diagnostics.
**Math and numeric conventions are carried over verbatim** from the frozen 2026-09-07 pilot and
2026-09-08 P0/P2 run; the only changes are packaging (shared `core`, CLI paths, tests).
This package **never downloads anything** and **does not bundle raw data**.

## Install
```bash
pip install -r requirements.txt      # numpy>=1.26.4  pandas>=2.3.3  scipy>=1.15.3
```
No new dependency was introduced. `tabulate` is deliberately *not* required (tables use
`DataFrame.to_string`). `run_p2.py` reports peak RSS via the stdlib `resource` module (POSIX-only).

## Commands
```bash
export PYTHONPATH=.
python p0_admission.py   --data-dir DATA --output-dir OUT
python run_p2.py         --data-dir DATA --output-dir OUT [--smoke]
python summarize_p2.py   --results OUT/results_p2.csv --output-dir OUT
python verify_alignment.py --data-dir DATA --baseline-dir BASE --output-dir OUT [--max-rows N]
```
`--help` works **without any data present**. Full help text is captured in `cli_help.txt`.
Defaults are relative (`./data`, `./out`, `./baseline`) and contain no absolute machine paths.

## Inputs
`--data-dir` must contain `pilot_counts.npz` and `pilot_cells.csv`. If they are absent the error
message prints the complete schema plus the official source (GEO **GSE164378**) — see
`compstate/io_data.py:SCHEMA_DOC`. These files are an **external input premise**; redistribution
terms for GSE164378 are **UNVERIFIED**.

`verify_alignment.py` additionally needs a frozen baseline `results_compstate.csv` in
`--baseline-dir` (~22 MB, also an external input; not bundled). It is only a regression check —
`p0_admission.py` / `run_p2.py` / `summarize_p2.py` do not need it.

## Tests (synthetic only, no real data)
```bash
python -m unittest discover -s tests -t .
```
11 tests covering: homogeneous-constraint residual within tolerance; scale invariance under
`y -> alpha*y` (evidence the true cell count **N is never used**); constraint-row homogeneity;
θ **8-dim** vs 4-dim (equal only when π is fixed, to the constraint residual; must differ when π is
free); `SPOTKEY` containing `delta`/`eta` and grid points not collapsing; identical reference
formula across arms; reference/generation pools disjoint per donor; declared privileged arms;
missing-input and cell-order-mismatch errors.

## Interpretation boundaries (do not drop)
- **Privileged arms** — `D1_truepi_rna`, `D1_truepi_joint` (given true π, **proportions only, never N**)
  and `D2_same_donor` (given the target donor's own reference pool). These are **not deployment
  performance and not an upper bound**.
- **Control arms** — `D2_cross_full` (baseline) and `D2_cross_matched` (size-matched control).
  `D3_stage1_joint` / `D3_stage2_rna` receive **no** privileged information.
- `D2_same_donor` vs `D2_cross_matched` shows only that the **reference-source condition changed**;
  donor identity is not separated from batch/capture/composition, so results alone must not be
  attributed to biological donor identity.
- Free-fit arms record `constraint_resid = 0.0` as a **placeholder**, not "constraint satisfied".
- Nominal arm only (`depth_thin=1.0`, `adt_dropout=0.0`); 2 donors per split; seeds and pseudo-spots
  are **technical replicates**, so results are descriptive — no margin, no significance test.
- `summary_s_support.csv` `rows`/`s_defined` counts **coarse-type evaluations** (spots × 4), not spots.
