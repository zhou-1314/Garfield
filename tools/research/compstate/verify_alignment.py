#!/usr/bin/env python3
"""Parity check against a FROZEN baseline run (regression guard for the port).

Regenerates the nominal spots with the frozen rng scheme, recomputes the baseline
crc32 cell-index hash convention, joins on the baseline's full row key
(seed,split,donor,scenario,delta,eta,spot) and measures:
  (a) byte-identical cell selection,
  (b) whether the original M4 (joint flex, cross_full reference) recomputed on the
      regenerated spot reproduces the stored pi_hat / s_hat.
The baseline stores 4 decimals, so ~5e-5 is the storage-precision floor.
Nothing is overwritten; this only reads the baseline.
"""
import argparse, sys

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./data", help="raw inputs")
    p.add_argument("--baseline-dir", default="./baseline",
                   help="directory holding the frozen results_compstate.csv")
    p.add_argument("--baseline-csv", default="results_compstate.csv")
    p.add_argument("--output-dir", default="./out")
    p.add_argument("--max-rows", type=int, default=0,
                   help="limit baseline rows compared (0 = all); use a small number for a quick check")
    p.add_argument("--tol", type=float, default=1e-4, help="tolerance for pi/s agreement")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    from compstate.core import set_threads; set_threads(4)
    import os, json, zlib
    import numpy as np, pandas as pd
    from compstate.config import CFG, CO, SUB, SUB_OF_C, SPLIT
    from compstate import core, io_data
    base = os.path.join(args.baseline_dir, args.baseline_csv)
    if not os.path.exists(base):
        raise SystemExit(f"baseline CSV not found: {base}\n"
                         "Pass --baseline-dir pointing at the frozen pilot analysis directory. "
                         "This file is an EXTERNAL INPUT and is not bundled (it is ~22 MB).")
    os.makedirs(args.output_dir, exist_ok=True)
    RNA, ADT, cells = io_data.load(args.data_dir)
    nC = len(cells)
    sub_idx = cells["celltype.l2"].map({s: i for i, s in enumerate(SUB)}).to_numpy()
    donor = cells["donor"].to_numpy()
    split_of = {d: k for k, v in SPLIT.items() for d in v}
    split = np.array([split_of[d] for d in donor])
    is_ref = core.make_pools(donor, sub_idx, nC)
    R_rna, R_adt, _ = core.build_ref(RNA, ADT, sub_idx, (split == "train") & is_ref)
    gene_sel = core.select_features(R_rna)
    refd = dict(Rr=R_rna[gene_sel], Ra=R_adt)
    RNA_SEL = RNA[gene_sel].copy()
    CRT, CAT = RNA.sum(0), ADT.sum(0)

    regen = {}
    for seed in CFG["seeds"]:
        for spl in ("val", "test"):
            genm = (split == spl) & (~is_ref); dh = sorted(set(donor[genm]))
            pools = {d: {j: np.where(genm & (donor == d) & (sub_idx == j))[0]
                         for j in range(len(SUB))} for d in dh}
            for scn in ("S0", "S1", "S2", "S3"):
                for (delta, eta) in core.grid_points(scn):
                    rng = core.spot_rng(seed, spl, scn, delta, eta)
                    pi_set, s_set = core.truth(delta, eta)
                    for k in range(CFG["n_spots"]):
                        d_k = dh[k % len(dh)]
                        got = core.make_spot(pools[d_k], pi_set, s_set, rng, RNA_SEL, ADT,
                                             sub_idx, CRT, CAT)
                        if got is None: continue
                        (r, a), _, _, _, idx = got
                        regen[(seed, spl, d_k, scn, round(delta, 10), round(eta, 10), k)] = dict(
                            crc32=zlib.crc32(",".join(map(str, sorted(idx.tolist()))).encode()),
                            r=r, a=a)

    old = pd.read_csv(base)
    nom = old[(old.depth_thin == 1.0) & (old.adt_dropout == 0.0) & (old.model == "M4")]
    if args.max_rows: nom = nom.head(args.max_rows)
    match = mismatch = missing = 0; pi_max = s_max = 0.0; ex = []
    for _, row in nom.iterrows():
        key = (int(row["seed"]), row["split"], row["donor"], row["scenario"],
               round(float(row["delta"]), 10), round(float(row["eta"]), 10), int(row["spot"]))
        g = regen.get(key)
        if g is None: missing += 1; continue
        if int(row["cell_idx_hash"]) == g["crc32"]:
            match += 1
            out = core.solve_free(refd, g["r"], g["a"], "joint")
            po = np.array([float(x) for x in str(row["pi_hat"]).split(";")])
            so = np.array([float(x) for x in str(row["s_hat"]).split(";")])
            pi_max = max(pi_max, np.abs(out["pi"]-po).max())
            s_max = max(s_max, np.abs(out["s"]-so).max())
        else:
            mismatch += 1
            if len(ex) < 5: ex.append({"key": str(key), "baseline": int(row["cell_idx_hash"]),
                                       "regen": g["crc32"]})
    ok = mismatch == 0 and missing == 0 and pi_max < args.tol and s_max < args.tol
    res = dict(baseline_rows_compared=int(len(nom)), regenerated_spots=len(regen),
               cell_idx_hash_match=match, cell_idx_hash_mismatch=mismatch,
               key_missing_in_regen=missing, max_abs_pi_diff=float(pi_max),
               max_abs_s_diff=float(s_max), tol=args.tol, mismatch_examples=ex,
               verdict="ALIGNED" if ok else "NOT ALIGNED",
               note="baseline stores 4 decimals => ~5e-5 is the storage-precision floor")
    json.dump(res, open(f"{args.output_dir}/alignment_vs_baseline.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
