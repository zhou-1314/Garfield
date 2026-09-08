#!/usr/bin/env python3
"""P0 data admission checks.  Reports PASS / FAIL / UNKNOWN / NOTE.
UNKNOWN is recorded verbatim and never upgraded to PASS."""
import argparse, sys

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./data", help="directory with the raw inputs")
    p.add_argument("--output-dir", default="./out", help="where p0_admission.json is written")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    import os, json, hashlib
    import numpy as np, pandas as pd
    from compstate.config import SPLIT, COARSE, CO, SUB, CFG
    from compstate import core, io_data
    os.makedirs(args.output_dir, exist_ok=True)
    R = {}
    def rec(k, st, det): R[k] = {"status": st, "detail": det}; print(f"[{st}] {k}: {det}")

    RNA, ADT, cells = io_data.load(args.data_dir)
    nC = len(cells)
    rec("A.inputs_loaded", "PASS", f"n_cells={nC} RNA{RNA.shape} ADT{ADT.shape}")
    rec("B.cell_order", "PASS", "npz['cells'] == csv['cell_id'] verified during load")
    rec("B.cell_id_unique", "PASS" if cells.cell_id.duplicated().sum() == 0 else "FAIL",
        f"duplicated cell_id = {int(cells.cell_id.duplicated().sum())}")
    for nm, M in (("RNA", RNA), ("ADT", ADT)):
        neg = int((M < 0).sum()); nonint = int((M != np.floor(M)).sum())
        rec(f"C.{nm}_raw_nonneg_integer", "PASS" if neg == 0 and nonint == 0 else "FAIL",
            f"neg={neg} non_integer={nonint} max={M.max()}")
    donor = cells["donor"].to_numpy()
    assign = {d: k for k, v in SPLIT.items() for d in v}
    unassigned = sorted(set(donor) - set(assign))
    rec("D.donor_split_disjoint", "PASS" if not unassigned else "FAIL",
        f"donors={sorted(set(donor))} unassigned={unassigned}")
    sub_idx = cells["celltype.l2"].map({s: i for i, s in enumerate(SUB)}).to_numpy()
    is_ref = core.make_pools(donor, sub_idx, nC)
    rec("E.ref_gen_pool_disjoint", "PASS",
        f"ref={int(is_ref.sum())} gen={int((~is_ref).sum())} overlap=0 by construction "
        f"(seed={CFG['ref_seed']}, frac={CFG['ref_pool_frac']}, drawn within donor x subtype)")
    split = np.array([assign[d] for d in donor])
    train_ref = (split == "train") & is_ref
    prov = {}
    for j, s in enumerate(SUB):
        ids = sorted(cells.cell_id.to_numpy()[train_ref & (sub_idx == j)].tolist())
        prov[s] = {"n_cells": len(ids),
                   "cell_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    rec("G.reference_provenance", "PASS", prov)
    rec("G.hierarchy_hash", "PASS", hashlib.sha256(
        json.dumps({c: COARSE[c] for c in CO}, sort_keys=True).encode()).hexdigest())
    tot = sum(int(round(0.25*CFG["n_cells_per_spot"])) for _ in CO)
    rec("H.S0_actual_cells_per_spot", "PASS",
        f"nominal n_cells_per_spot={CFG['n_cells_per_spot']} but banker's rounding gives "
        f"ACTUAL total={tot} at S0. Report {tot}, NEVER {CFG['n_cells_per_spot']}.")
    rec("I.source_licence", "UNKNOWN",
        "GSE164378 redistribution terms NOT verified; no licence file ships with the raw inputs. "
        "Recorded as UNKNOWN, not asserted as permissive.")
    n_fail = sum(1 for v in R.values() if v["status"] == "FAIL")
    n_unk = sum(1 for v in R.values() if v["status"] == "UNKNOWN")
    summ = {"n_checks": len(R), "n_fail": n_fail, "n_unknown": n_unk,
            "admission": "PASS_WITH_UNKNOWNS" if n_fail == 0 else "FAIL"}
    json.dump({"summary": summ, "checks": R},
              open(f"{args.output_dir}/p0_admission.json", "w"), indent=1, default=str)
    print("\n==== P0 SUMMARY ====", json.dumps(summ))
    return 0 if n_fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
