#!/usr/bin/env python3
"""PORTABLE rewrite of reproduce_p1_eval.py (original audited script SHA 73e15773d306db7ec7d648fac15bc52577648fe2b10121a637e40a1a466d600e).

reproduce_p1_eval.py -- Independent P1 Baseline Audit & Post-Processing Script

Task: R7-EXEC-P0P2-20260908
Agent: spamot-5 (Independent Auditor)

This script performs reproducible post-processing and metric evaluation of the P1 baseline
pilot results (results_compstate.csv from spamot-2).

Key Capabilities:
1. Input verification via SHA-256 and schema checks.
2. Rowkey-based 1:1 matching to construct Stacked model (M4_pi + M3_s).
3. 8-dimensional theta reconstruction (theta_true and theta_hat) for M1-M4 and Stacked:
   - For M1/M2: within-type state s is not estimated by design (s_denom = 0, s_mae = NA);
     theta_hat is reconstructed via frozen baseline s0 = 0.5 (two subtypes per coarse type).
   - For M3/M4: theta_hat is reconstructed from estimated pi and s.
   - For Stacked: theta_hat is reconstructed from M4 pi and M3 s.
4. Mathematical & architectural invariant tests:
   - Simplex sum to 1 (pi and theta)
   - Tree consistency (theta_2c + theta_2c+1 == pi_c)
   - Exact fixed-point recovery for fixed s0 (theta_MAE == 0.5 * pi_MAE when s_true == 0.5)
   - Key uniqueness and 1:1 pairing
   - Explicit denominator tracking
5. Metric aggregation across two formal calibers:
   - Caliber 1 (Hierarchical Macro): spot -> seed mean -> scene/grid mean -> donor macro mean
   - Caliber 2 (Pooled Row-Average): direct average across all rows in stratum
6. Complete CSV exports:
   - results_p1_per_row.csv (79,200 rows)
   - tables_p1_nominal_test.csv
   - tables_p1_nominal_val.csv
   - tables_p1_all_arms.csv
   - tables_p1_donor_summary.csv
   - tables_p1_calibers_comparison.csv
"""

import sys
import os
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

# Expected input file hashes (frozen 2026-09-07 pilot artifacts)
EXPECTED_HASHES = {
    "results_compstate.csv": "9a1c7d8efd80dd4ac384311edfa5a7a6396647a74646a3caaedd942e21feee18",
    "config_frozen.json": "f10f87fad7713fb0f2d24d69eb3ebeb8c6f8f92002679097d8809e5dfedf9ae8",
    "summary_compstate.json": "b2921f3db4e562cee601e85cdc85a738bf8877c85ec225a95dacd62c1815b120"
}

ROW_KEYS = ["seed", "split", "donor", "scenario", "delta", "eta", "depth_thin", "adt_dropout", "spot"]
COARSE_ORDER = ["B", "CD4 T", "CD8 T", "Mono"]
SUBTYPE_ORDER = [
    "B naive", "B memory",
    "CD4 Naive", "CD4 TCM",
    "CD8 TEM", "CD8 Naive",
    "CD14 Mono", "CD16 Mono"
]


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs(input_dir: str):
    print("=== [1/5] Verifying Input Files and Provenance ===")
    for fname, expected_hash in EXPECTED_HASHES.items():
        fpath = os.path.join(input_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Missing required input file: {fpath}")
        actual_hash = sha256_file(fpath)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA256 mismatch for {fname}!\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )
        print(f"  [PASS] {fname} (SHA256: {actual_hash[:16]}...)")


def parse_semicolon_vector(s: str) -> np.ndarray:
    if pd.isna(s) or s == "NA" or s == "" or s is None:
        return np.array([])
    return np.array([float(x) for x in str(s).split(";")])


def format_semicolon_vector(arr: np.ndarray) -> str:
    if arr is None or len(arr) == 0 or np.isnan(arr).all():
        return "NA"
    return ";".join(f"{x:.4f}" for x in arr)


def build_augmented_dataset(input_dir: str) -> pd.DataFrame:
    print("\n=== [2/5] Loading Data and Constructing Stacked & Theta Estimators ===")
    csv_path = os.path.join(input_dir, "results_compstate.csv")
    df = pd.read_csv(csv_path)
    print(f"  Loaded results_compstate.csv: {len(df)} rows, {len(df.columns)} columns")

    # Verify model presence and count
    models = ["M1", "M2", "M3", "M4"]
    for m in models:
        cnt = (df["model"] == m).sum()
        if cnt != 15840:
            raise ValueError(f"Expected 15,840 rows for model {m}, found {cnt}")

    # Separate into individual model dataframes indexed by ROW_KEYS
    m_dfs = {}
    for m in models:
        sub = df[df["model"] == m].copy().set_index(ROW_KEYS)
        if len(sub) != len(sub.index.drop_duplicates()):
            raise ValueError(f"Duplicate row keys detected in model {m}!")
        m_dfs[m] = sub

    # Check that indexes match exactly across models
    for m in ["M2", "M3", "M4"]:
        if not m_dfs["M1"].index.equals(m_dfs[m].index):
            raise ValueError(f"Row keys between M1 and {m} are not in identical alignment!")

    # Verify that spot properties match across models
    for m in ["M2", "M3", "M4"]:
        if not (m_dfs["M1"]["cell_idx_hash"] == m_dfs[m]["cell_idx_hash"]).all():
            raise ValueError(f"cell_idx_hash mismatch between M1 and {m}!")
        if not (m_dfs["M1"]["pi_true"] == m_dfs[m]["pi_true"]).all():
            raise ValueError(f"pi_true mismatch between M1 and {m}!")
        if not (m_dfs["M1"]["s_true"] == m_dfs[m]["s_true"]).all():
            raise ValueError(f"s_true mismatch between M1 and {m}!")

    print("  [PASS] All 4 baseline models share identical 1:1 rowkey alignment and spot truth.")

    # Construct Stacked model: M4 pi + M3 s (matched strictly 1:1 on row key)
    stacked = m_dfs["M4"].copy()
    stacked["model"] = "Stacked"
    stacked["s_hat"] = m_dfs["M3"]["s_hat"].values
    stacked["s_mae"] = m_dfs["M3"]["s_mae"].values
    stacked["s_rmse"] = m_dfs["M3"]["s_rmse"].values
    stacked["s_is_constant"] = 0
    stacked["s_drift_from_s0"] = m_dfs["M3"]["s_drift_from_s0"].values

    all_models = {
        "M1": m_dfs["M1"],
        "M2": m_dfs["M2"],
        "M3": m_dfs["M3"],
        "M4": m_dfs["M4"],
        "Stacked": stacked
    }

    # Reconstruct theta_true and theta_hat for each model
    processed_dfs = []
    for model_name, mdf in all_models.items():
        mdf_proc = mdf.reset_index()
        n_rows = len(mdf_proc)

        theta_true_list = []
        theta_hat_list = []
        theta_mae_list = np.zeros(n_rows, dtype=np.float64)
        theta_rmse_list = np.zeros(n_rows, dtype=np.float64)
        s_denom_list = np.zeros(n_rows, dtype=np.int32)

        # Parse arrays in bulk
        pi_true_mat = np.array([parse_semicolon_vector(s) for s in mdf_proc["pi_true"]])
        s_true_mat = np.array([parse_semicolon_vector(s) for s in mdf_proc["s_true"]])
        pi_hat_mat = np.array([parse_semicolon_vector(s) for s in mdf_proc["pi_hat"]])

        # theta_true: shape (n_rows, 8)
        # coarse types 0..3 -> subtypes 2c, 2c+1
        theta_true_mat = np.zeros((n_rows, 8), dtype=np.float64)
        for c in range(4):
            theta_true_mat[:, 2 * c] = pi_true_mat[:, c] * s_true_mat[:, c]
            theta_true_mat[:, 2 * c + 1] = pi_true_mat[:, c] * (1.0 - s_true_mat[:, c])

        theta_hat_mat = np.zeros((n_rows, 8), dtype=np.float64)
        if model_name in ["M1", "M2"]:
            # Fixed models: s is not estimated by design; reconstruct with s0 = 0.5
            s_denom_list[:] = 0
            for c in range(4):
                theta_hat_mat[:, 2 * c] = pi_hat_mat[:, c] * 0.5
                theta_hat_mat[:, 2 * c + 1] = pi_hat_mat[:, c] * 0.5
        else:
            # Flex models (M3, M4, Stacked): reconstruct with estimated s
            s_hat_mat = np.array([parse_semicolon_vector(s) for s in mdf_proc["s_hat"]])
            s_denom_list[:] = 1
            for c in range(4):
                theta_hat_mat[:, 2 * c] = pi_hat_mat[:, c] * s_hat_mat[:, c]
                theta_hat_mat[:, 2 * c + 1] = pi_hat_mat[:, c] * (1.0 - s_hat_mat[:, c])

        # Compute theta MAE and RMSE per spot across 8 subtypes
        diff = theta_hat_mat - theta_true_mat
        theta_mae_list = np.mean(np.abs(diff), axis=1)
        theta_rmse_list = np.sqrt(np.mean(diff ** 2, axis=1))

        # Format string vectors for serialization
        theta_true_str = [format_semicolon_vector(row) for row in theta_true_mat]
        theta_hat_str = [format_semicolon_vector(row) for row in theta_hat_mat]

        mdf_proc["theta_true"] = theta_true_str
        mdf_proc["theta_hat"] = theta_hat_str
        mdf_proc["theta_mae"] = theta_mae_list
        mdf_proc["theta_rmse"] = theta_rmse_list
        mdf_proc["s_denom"] = s_denom_list

        # Ensure s_mae is NA / NaN for M1, M2
        if model_name in ["M1", "M2"]:
            mdf_proc["s_mae"] = np.nan
            mdf_proc["s_rmse"] = np.nan
            mdf_proc["s_hat"] = "NA"

        processed_dfs.append(mdf_proc)

    combined_df = pd.concat(processed_dfs, ignore_index=True)
    print(f"  Constructed augmented dataset: {len(combined_df)} total rows across 5 models.")
    return combined_df


def run_invariant_tests(df: pd.DataFrame):
    print("\n=== [3/5] Running Invariant Verification & Consistency Checks ===")

    # Test 1: Simplex Sum-to-One
    print("  Test 1: Checking simplex sum-to-one constraints...")
    for idx, row in df.iloc[::250].iterrows():
        pi_t = parse_semicolon_vector(row["pi_true"])
        th_t = parse_semicolon_vector(row["theta_true"])
        pi_h = parse_semicolon_vector(row["pi_hat"])
        th_h = parse_semicolon_vector(row["theta_hat"])

        assert np.isclose(np.sum(pi_t), 1.0, atol=1e-3), f"pi_true does not sum to 1 at row {idx}"
        assert np.isclose(np.sum(th_t), 1.0, atol=1e-3), f"theta_true does not sum to 1 at row {idx}"
        assert np.isclose(np.sum(pi_h), 1.0, atol=1e-3), f"pi_hat does not sum to 1 at row {idx}"
        assert np.isclose(np.sum(th_h), 1.0, atol=1e-3), f"theta_hat does not sum to 1 at row {idx}"
    print("    [PASS] Simplex sum-to-1 holds for pi and theta across ground truth and predictions.")

    # Test 2: Tree Consistency (Sum of subtypes equals coarse fraction)
    print("  Test 2: Checking tree consistency (theta[2c] + theta[2c+1] == pi[c])...")
    for idx, row in df.iloc[::250].iterrows():
        pi_t = parse_semicolon_vector(row["pi_true"])
        th_t = parse_semicolon_vector(row["theta_true"])
        pi_h = parse_semicolon_vector(row["pi_hat"])
        th_h = parse_semicolon_vector(row["theta_hat"])

        for c in range(4):
            assert np.isclose(th_t[2*c] + th_t[2*c+1], pi_t[c], atol=1e-3), \
                f"Tree consistency failure in truth at row {idx}, type {c}"
            assert np.isclose(th_h[2*c] + th_h[2*c+1], pi_h[c], atol=1e-3), \
                f"Tree consistency failure in hat at row {idx}, type {c}"
    print("    [PASS] Tree consistency holds across all examined spots and models.")

    # Test 3: Fixed s0 Exact Fixed-Point Recovery Theorem
    # On spots where s_true is exactly 0.5 for all coarse types (e.g., in S0 and delta=0 grid points):
    # theta_hat - theta_true = 0.5 * (pi_hat - pi_true).
    # Consequently, theta_MAE == 0.5 * pi_MAE identically!
    # Note: in S1 with delta = +/-0.15, discrete 50-cell sampling yields 5 cells for one type,
    # where round(0.5*5) = 2 cells, giving discrete s_true = 2/5 = 0.4000.
    print("  Test 3: Checking exact fixed s0 theorem (theta_MAE == 0.5 * pi_MAE when s_true == 0.5)...")
    for m in ["M1", "M2"]:
        sub = df[(df["model"] == m) & (df["s_true"] == "0.5000;0.5000;0.5000;0.5000")]
        diff = np.abs(sub["theta_mae"] - 0.5 * sub["pi_mae"])
        max_diff = np.max(diff)
        assert max_diff < 1e-4, f"Fixed s0 recovery theorem violation in {m}: max diff = {max_diff}"
    print("    [PASS] Fixed s0 recovery holds: theta_MAE == 0.5 * pi_MAE on all spots with s_true==0.5.")

    # Test 4: Denominator Accounting and NA Policy
    print("  Test 4: Checking denominator and NA policies...")
    m1_m2 = df[df["model"].isin(["M1", "M2"])]
    assert (m1_m2["s_denom"] == 0).all(), "M1/M2 s_denom must be 0"
    assert m1_m2["s_mae"].isna().all(), "M1/M2 s_mae must be NaN"

    flex = df[df["model"].isin(["M3", "M4", "Stacked"])]
    assert (flex["s_denom"] == 1).all(), "Flex model s_denom must be 1"
    assert flex["s_mae"].notna().all(), "Flex model s_mae must not be NaN"
    print("    [PASS] Denominators and NA policies verified.")


def compute_caliber_metrics(df_sub: pd.DataFrame, scenario: str, model: str) -> dict:
    """Computes metrics under both Caliber 1 and Caliber 2 for a specific scenario & model."""
    g = df_sub[(df_sub["scenario"] == scenario) & (df_sub["model"] == model)]
    n_used = len(g)
    if n_used == 0:
        return {}

    # Caliber 2: Pooled row-average
    cal2_pi_mae = float(g["pi_mae"].mean())
    cal2_pi_rmse = float(g["pi_rmse"].mean())
    cal2_s_mae = float(g["s_mae"].mean()) if model not in ["M1", "M2"] else np.nan
    cal2_s_denom = int(g["s_denom"].sum())
    cal2_th_mae = float(g["theta_mae"].mean())
    cal2_th_rmse = float(g["theta_rmse"].mean())

    # Caliber 1: Hierarchical Macro
    # Level 1: mean over spots per (donor, delta, eta, seed)
    g1 = g.groupby(["donor", "delta", "eta", "seed"])[["pi_mae", "pi_rmse", "s_mae", "theta_mae", "theta_rmse"]].mean().reset_index()
    # Level 2: mean over seeds per (donor, delta, eta)
    g2 = g1.groupby(["donor", "delta", "eta"])[["pi_mae", "pi_rmse", "s_mae", "theta_mae", "theta_rmse"]].mean().reset_index()
    # Level 3: mean over grid points (scenes) per donor
    g3 = g2.groupby(["donor"])[["pi_mae", "pi_rmse", "s_mae", "theta_mae", "theta_rmse"]].mean().reset_index()
    # Level 4: macro mean over donors
    cal1_pi_mae = float(g3["pi_mae"].mean())
    cal1_pi_rmse = float(g3["pi_rmse"].mean())
    cal1_s_mae = float(g3["s_mae"].mean()) if model not in ["M1", "M2"] else np.nan
    cal1_th_mae = float(g3["theta_mae"].mean())
    cal1_th_rmse = float(g3["theta_rmse"].mean())

    return {
        "scenario": scenario,
        "model": model,
        "n_used": n_used,
        "s_denom": cal2_s_denom,
        # Caliber 1
        "cal1_pi_mae": cal1_pi_mae,
        "cal1_pi_rmse": cal1_pi_rmse,
        "cal1_s_mae": cal1_s_mae,
        "cal1_th_mae": cal1_th_mae,
        "cal1_th_rmse": cal1_th_rmse,
        # Caliber 2
        "cal2_pi_mae": cal2_pi_mae,
        "cal2_pi_rmse": cal2_pi_rmse,
        "cal2_s_mae": cal2_s_mae,
        "cal2_th_mae": cal2_th_mae,
        "cal2_th_rmse": cal2_th_rmse,
    }


def compute_overall_caliber_metrics(df_sub: pd.DataFrame, model: str) -> dict:
    """Computes overall aggregate metrics across all scenarios under Caliber 1 vs Caliber 2."""
    g = df_sub[df_sub["model"] == model]
    n_used = len(g)

    # Caliber 2: Direct row average over all 1,320 rows
    cal2_pi_mae = float(g["pi_mae"].mean())
    cal2_pi_rmse = float(g["pi_rmse"].mean())
    cal2_s_mae = float(g["s_mae"].mean()) if model not in ["M1", "M2"] else np.nan
    cal2_s_denom = int(g["s_denom"].sum())
    cal2_th_mae = float(g["theta_mae"].mean())
    cal2_th_rmse = float(g["theta_rmse"].mean())

    # Caliber 1: Equal weighting of scenarios and donors
    # 1. Spot -> seed mean
    g1 = g.groupby(["donor", "scenario", "delta", "eta", "seed"])[["pi_mae", "pi_rmse", "s_mae", "theta_mae", "theta_rmse"]].mean().reset_index()
    # 2. Seed -> grid point mean
    g2 = g1.groupby(["donor", "scenario", "delta", "eta"])[["pi_mae", "pi_rmse", "s_mae", "theta_mae", "theta_rmse"]].mean().reset_index()
    # 3. Grid point -> scenario mean per donor
    g3 = g2.groupby(["donor", "scenario"])[["pi_mae", "pi_rmse", "s_mae", "theta_mae", "theta_rmse"]].mean().reset_index()
    # 4. Equal weight over scenarios per donor
    g4 = g3.groupby(["donor"])[["pi_mae", "pi_rmse", "s_mae", "theta_mae", "theta_rmse"]].mean().reset_index()
    # 5. Macro mean over donors
    cal1_pi_mae = float(g4["pi_mae"].mean())
    cal1_pi_rmse = float(g4["pi_rmse"].mean())
    cal1_s_mae = float(g4["s_mae"].mean()) if model not in ["M1", "M2"] else np.nan
    cal1_th_mae = float(g4["theta_mae"].mean())
    cal1_th_rmse = float(g4["theta_rmse"].mean())

    return {
        "scenario": "ALL_SCENARIOS",
        "model": model,
        "n_used": n_used,
        "s_denom": cal2_s_denom,
        "cal1_pi_mae": cal1_pi_mae,
        "cal1_pi_rmse": cal1_pi_rmse,
        "cal1_s_mae": cal1_s_mae,
        "cal1_th_mae": cal1_th_mae,
        "cal1_th_rmse": cal1_th_rmse,
        "cal2_pi_mae": cal2_pi_mae,
        "cal2_pi_rmse": cal2_pi_rmse,
        "cal2_s_mae": cal2_s_mae,
        "cal2_th_mae": cal2_th_mae,
        "cal2_th_rmse": cal2_th_rmse,
    }


def generate_and_export_tables(df: pd.DataFrame, output_dir: str):
    print("\n=== [4/5] Generating Metric Tables and Post-Processed Datasets ===")
    models = ["M1", "M2", "M3", "M4", "Stacked"]
    scenarios = ["S0", "S1", "S2", "S3"]

    # 1. Export results_p1_per_row.csv
    per_row_path = os.path.join(output_dir, "results_p1_per_row.csv")
    cols_order = [
        "seed", "split", "donor", "scenario", "delta", "eta", "depth_thin", "adt_dropout", "spot",
        "model", "cell_idx_hash", "n_cells", "s_is_constant",
        "pi_mae", "pi_rmse", "s_mae", "s_rmse", "s_denom", "theta_mae", "theta_rmse",
        "pi_drift_from_pi0", "s_drift_from_s0",
        "pi_true", "pi_hat", "s_true", "s_hat", "theta_true", "theta_hat",
        "coef_share", "m_rna_true", "m_adt_true"
    ]
    df[cols_order].to_csv(per_row_path, index=False)
    print(f"  [EXPORT] results_p1_per_row.csv ({len(df)} rows)")

    # 2. Nominal Test Table (split == test, depth_thin == 1.0, adt_dropout == 0.0)
    test_nom = df[(df["split"] == "test") & (df["depth_thin"] == 1.0) & (df["adt_dropout"] == 0.0)]
    test_rows = []
    for scn in scenarios:
        for m in models:
            test_rows.append(compute_caliber_metrics(test_nom, scn, m))
    df_test_nom = pd.DataFrame(test_rows)
    df_test_nom.to_csv(os.path.join(output_dir, "tables_p1_nominal_test.csv"), index=False)
    print(f"  [EXPORT] tables_p1_nominal_test.csv ({len(df_test_nom)} rows)")

    # 3. Nominal Val Table (split == val, depth_thin == 1.0, adt_dropout == 0.0)
    val_nom = df[(df["split"] == "val") & (df["depth_thin"] == 1.0) & (df["adt_dropout"] == 0.0)]
    val_rows = []
    for scn in scenarios:
        for m in models:
            val_rows.append(compute_caliber_metrics(val_nom, scn, m))
    df_val_nom = pd.DataFrame(val_rows)
    df_val_nom.to_csv(os.path.join(output_dir, "tables_p1_nominal_val.csv"), index=False)
    print(f"  [EXPORT] tables_p1_nominal_val.csv ({len(df_val_nom)} rows)")

    # 4. Calibers Comparison Table (Side-by-Side: S0-S3 + Overall across scenarios)
    cal_rows = []
    for spl_name, spl_df in [("test", test_nom), ("val", val_nom)]:
        for scn in scenarios:
            for m in models:
                rec = compute_caliber_metrics(spl_df, scn, m)
                rec["split"] = spl_name
                cal_rows.append(rec)
        for m in models:
            rec = compute_overall_caliber_metrics(spl_df, m)
            rec["split"] = spl_name
            cal_rows.append(rec)
    df_calibers = pd.DataFrame(cal_rows)
    df_calibers.to_csv(os.path.join(output_dir, "tables_p1_calibers_comparison.csv"), index=False)
    print(f"  [EXPORT] tables_p1_calibers_comparison.csv ({len(df_calibers)} rows)")

    # 5. All Arms Table (Breakdown across all 6 noise arms x 4 scenarios x 5 models x 2 splits)
    all_arms_rows = []
    for (spl, scn, thin, drop, m), g in df.groupby(["split", "scenario", "depth_thin", "adt_dropout", "model"]):
        s_val = float(g["s_mae"].mean()) if m not in ["M1", "M2"] else np.nan
        s_denom = int(g["s_denom"].sum())
        all_arms_rows.append({
            "split": spl,
            "scenario": scn,
            "depth_thin": thin,
            "adt_dropout": drop,
            "model": m,
            "n_used": len(g),
            "pi_mae": round(float(g["pi_mae"].mean()), 4),
            "pi_rmse": round(float(g["pi_rmse"].mean()), 4),
            "s_mae": round(s_val, 4) if not np.isnan(s_val) else "NA",
            "s_rmse": round(float(g["s_rmse"].mean()), 4) if m not in ["M1", "M2"] else "NA",
            "s_denom": s_denom,
            "theta_mae": round(float(g["theta_mae"].mean()), 4),
            "theta_rmse": round(float(g["theta_rmse"].mean()), 4),
        })
    df_all_arms = pd.DataFrame(all_arms_rows)
    df_all_arms.to_csv(os.path.join(output_dir, "tables_p1_all_arms.csv"), index=False)
    print(f"  [EXPORT] tables_p1_all_arms.csv ({len(df_all_arms)} rows)")

    # 6. Donor Macro Summary Table (Nominal arms, broken down per donor)
    donor_rows = []
    nom_all = df[(df["depth_thin"] == 1.0) & (df["adt_dropout"] == 0.0)]
    for (spl, dn, scn, m), g in nom_all.groupby(["split", "donor", "scenario", "model"]):
        s_val = float(g["s_mae"].mean()) if m not in ["M1", "M2"] else np.nan
        donor_rows.append({
            "split": spl,
            "donor": dn,
            "scenario": scn,
            "model": m,
            "n_used": len(g),
            "pi_mae": round(float(g["pi_mae"].mean()), 4),
            "s_mae": round(s_val, 4) if not np.isnan(s_val) else "NA",
            "s_denom": int(g["s_denom"].sum()),
            "theta_mae": round(float(g["theta_mae"].mean()), 4)
        })
    df_donor = pd.DataFrame(donor_rows)
    df_donor.to_csv(os.path.join(output_dir, "tables_p1_donor_summary.csv"), index=False)
    print(f"  [EXPORT] tables_p1_donor_summary.csv ({len(df_donor)} rows)")


def print_summary_report(output_dir: str):
    print("\n=== [5/5] Nominal Results Executive Summary (Caliber 1 / Caliber 2) ===")
    test_df = pd.read_csv(os.path.join(output_dir, "tables_p1_nominal_test.csv"))
    print("\n[TEST Nominal Arm: depth_thin=1.0, adt_dropout=0.0 (P4, P8; 3 seeds)]")
    print(f"{'Scenario':<10}{'Model':<10}{'N':<6}{'pi_MAE':<10}{'s_MAE':<12}{'s_denom':<10}{'theta_MAE':<12}{'theta_RMSE':<12}")
    print("-" * 82)
    for _, r in test_df.iterrows():
        s_str = f"{r['cal1_s_mae']:.4f}" if not pd.isna(r['cal1_s_mae']) else "NA (fixed)"
        print(f"{r['scenario']:<10}{r['model']:<10}{r['n_used']:<6}{r['cal1_pi_mae']:<10.4f}{s_str:<12}{r['s_denom']:<10}{r['cal1_th_mae']:<12.4f}{r['cal1_th_rmse']:<12.4f}")


def main():
    parser = argparse.ArgumentParser(description="P1 Baseline Evaluation & Post-Processing")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to directory containing input pilot files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./p1_out",
        help="Path to output directory for tables and post-processed data"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    verify_inputs(args.input_dir)
    df = build_augmented_dataset(args.input_dir)
    run_invariant_tests(df)
    generate_and_export_tables(df, args.output_dir)
    print_summary_report(args.output_dir)
    print("\n[SUCCESS] Independent P1 Baseline post-processing and metric export completed.")


if __name__ == "__main__":
    main()
