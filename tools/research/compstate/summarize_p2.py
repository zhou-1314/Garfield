#!/usr/bin/env python3
"""Summarise P2 results.  theta is EIGHT-dimensional: each coarse type contributes
theta1 = pi*s (first subtype) AND theta2 = pi*(1-s) (second subtype).  The results
CSV stores only the first; the second is reconstructed here.

Donor is the inference unit; seeds and pseudo-spots are TECHNICAL replicates.
"""
import argparse, sys

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="./out/results_p2.csv", help="results_p2.csv path")
    p.add_argument("--output-dir", default="./out", help="where summaries are written")
    return p

def summarize(df):
    import numpy as np, pandas as pd
    from compstate.config import SPOTKEY
    df = df.copy()
    df["theta1_true"] = df.pi_true*df.s_true;      df["theta1_hat"] = df.pi_hat*df.s_hat
    df["theta2_true"] = df.pi_true*(1-df.s_true);  df["theta2_hat"] = df.pi_hat*(1-df.s_hat)
    df["pi_ae"] = (df.pi_hat-df.pi_true).abs();    df["pi_se"] = (df.pi_hat-df.pi_true)**2
    df["s_ae"]  = (df.s_hat-df.s_true).abs()
    df["t1_ae"] = (df.theta1_hat-df.theta1_true).abs(); df["t1_se"] = (df.theta1_hat-df.theta1_true)**2
    df["t2_ae"] = (df.theta2_hat-df.theta2_true).abs(); df["t2_se"] = (df.theta2_hat-df.theta2_true)**2
    sp = df.groupby(SPOTKEY, as_index=False).agg(
        a1=("t1_ae","sum"), a2=("t2_ae","sum"), s1=("t1_se","sum"), s2=("t2_se","sum"),
        n_type=("coarse_type","size"), pi_mae=("pi_ae","mean"),
        pi_mse=("pi_se","mean"), s_mae=("s_ae","mean"))
    sp["theta8_mae"]  = (sp.a1+sp.a2)/(2*sp.n_type)
    sp["theta8_rmse"] = np.sqrt((sp.s1+sp.s2)/(2*sp.n_type))
    sp["pi_rmse"] = np.sqrt(sp.pi_mse)
    sp = sp.drop(columns=["a1","a2","s1","s2","pi_mse"])
    return df, sp

def main(argv=None):
    args = build_parser().parse_args(argv)
    import numpy as np, pandas as pd, os
    from compstate.config import PRIVILEGED
    if not os.path.exists(args.results):
        raise SystemExit(f"results file not found: {args.results}\nRun run_p2.py first.")
    os.makedirs(args.output_dir, exist_ok=True)
    df, sp = summarize(pd.read_csv(args.results))
    M = ["pi_mae","pi_rmse","s_mae","theta8_mae","theta8_rmse"]
    O = args.output_dir
    sp.to_csv(f"{O}/summary_spot_level.csv", index=False)
    a1 = sp.groupby(["arm","diagnostic","split","donor","scenario"], as_index=False)[M].mean()
    per_donor = a1.groupby(["arm","diagnostic","split","donor"], as_index=False)[M].mean()
    macro = per_donor.groupby(["arm","diagnostic","split"], as_index=False)[M].mean()
    macro_row = sp.groupby(["arm","diagnostic","split"], as_index=False)[M].mean()
    per_donor.to_csv(f"{O}/summary_per_donor.csv", index=False)
    macro.to_csv(f"{O}/summary_macro_scenario_equal_weight.csv", index=False)
    macro_row.to_csv(f"{O}/summary_macro_row_weight.csv", index=False)
    sp.groupby(["arm","diagnostic","split","scenario"], as_index=False)[M].mean() \
      .to_csv(f"{O}/summary_by_scenario.csv", index=False)
    bt = df.groupby(["arm","diagnostic","split","coarse_type"], as_index=False).agg(
        pi_mae=("pi_ae","mean"), s_mae=("s_ae","mean"),
        theta1_mae=("t1_ae","mean"), theta2_mae=("t2_ae","mean"),
        s_defined=("s_true", lambda x: int(x.notna().sum())), rows=("s_ae","size"))
    bt["theta_mae_bothsubtypes"] = (bt.theta1_mae+bt.theta2_mae)/2
    bt.to_csv(f"{O}/summary_by_type.csv", index=False)
    cr = df.groupby(["arm","diagnostic"], as_index=False).agg(
        resid_max=("constraint_resid","max"),
        resid_p99=("constraint_resid", lambda x: float(np.percentile(x, 99))))
    cr["constraint_applied"] = cr.diagnostic.isin(["D1","D3"]) & ~cr.arm.eq("D3_stage1_joint")
    cr["note"] = np.where(cr.constraint_applied, "fixed-pi constraint APPLIED",
        "FREE fit - the stored 0.0 is a PLACEHOLDER, NOT 'passed the fixed constraint'")
    cr.to_csv(f"{O}/summary_constraint_residual.csv", index=False)
    sup = df.groupby(["arm","split"], as_index=False).agg(
        rows=("s_ae","size"), s_defined=("s_true", lambda x: int(x.notna().sum())))
    sup["s_missing"] = sup.rows - sup.s_defined
    sup.to_csv(f"{O}/summary_s_support.csv", index=False)
    macro["privileged"] = macro.arm.isin(PRIVILEGED)
    print(macro.round(5).to_string(index=False))
    print("\nprivileged arms (NOT deployment performance, NOT an upper bound):", sorted(PRIVILEGED))
    print("units: summary_s_support rows = coarse-type evaluations = spots x 4; spots are in summary_spot_level")
    return 0

if __name__ == "__main__":
    sys.exit(main())
