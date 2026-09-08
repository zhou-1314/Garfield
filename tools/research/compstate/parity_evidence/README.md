# Parity evidence (regression only; nothing was overwritten)

1. **Portable package vs frozen 2026-09-08 results** — smoke run (`--smoke`, seed 1, 4 spots/grid)
   on the real local raw, joined to the frozen `results_p2.csv` on
   (arm, split, donor, scenario, delta, eta, seed, spot_id, coarse_type): **2,464/2,464 rows joined**,
   `max|Δ|` = **0.000e+00** for pi_hat / s_hat / theta_hat / pi_true / s_true / n_cells_true, and
   `cell_idx_sha256` identical on every row. **Exact reproduction.**
2. **Portable package vs frozen 2026-09-07 baseline** — `verify_alignment.py --max-rows 400`:
   400/400 crc32 cell-index hashes match, 0 mismatch, 0 missing;
   `max|Δ pi| = 5.00e-05`, `max|Δ s| = 4.98e-05` = the baseline's 4-decimal storage floor.
   Verdict **ALIGNED** (see `alignment_vs_baseline.json`).

Seeds, grid, models and scenarios were NOT expanded; original results were NOT overwritten.
