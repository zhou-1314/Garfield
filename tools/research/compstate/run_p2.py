#!/usr/bin/env python3
"""P2 diagnostics D1 / D2 / D3 -- portable CLI.  Nominal arm only (depth=1.0, dropout=0.0).

Privileged arms (NOT deployment performance, NOT an upper bound):
  D1_truepi_rna / D1_truepi_joint  -- given true pi (PROPORTIONS ONLY, never N)
  D2_same_donor                    -- given the target donor's own reference pool
Control arms: D2_cross_full (baseline), D2_cross_matched (size-matched control).
D3_stage1_joint / D3_stage2_rna receive NO privileged information.
"""
import argparse, sys

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./data",
                   help="directory holding pilot_counts.npz and pilot_cells.csv (default: ./data)")
    p.add_argument("--output-dir", default="./out", help="where results are written (default: ./out)")
    p.add_argument("--smoke", action="store_true", help="tiny run (1 seed, 4 spots per grid point)")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="override seeds (default 1 2 3)")
    p.add_argument("--n-spots", type=int, default=None, help="override spots per grid point (default 40)")
    p.add_argument("--n-genes", type=int, default=None, help="override selected genes (default 1500)")
    p.add_argument("--threads", type=int, default=4, help="BLAS/OMP thread cap (default 4)")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)          # --help never touches the data
    from compstate.core import set_threads
    set_threads(args.threads)                        # before numpy import below
    import os, json, csv, time, hashlib, collections
    import numpy as np, pandas as pd
    from compstate.config import CFG, CO, SUB, SUB_OF_C, SPLIT, ARMS, ROWCOLS, FAILCOLS
    from compstate import core, io_data

    t0 = time.time()
    if args.seeds:   CFG["seeds"] = args.seeds
    if args.n_spots: CFG["n_spots"] = args.n_spots
    if args.n_genes: CFG["n_genes"] = args.n_genes
    if args.smoke:   CFG["seeds"], CFG["n_spots"] = [1], 4
    os.makedirs(args.output_dir, exist_ok=True)

    RNA, ADT, cells = io_data.load(args.data_dir)
    nC = len(cells)
    sub_idx = cells["celltype.l2"].map({s: i for i, s in enumerate(SUB)}).to_numpy()
    donor = cells["donor"].to_numpy()
    split_of = {d: k for k, v in SPLIT.items() for d in v}
    unknown = sorted(set(donor) - set(split_of))
    if unknown:
        raise SystemExit(f"donor(s) {unknown} are not in the frozen split {SPLIT}")
    split = np.array([split_of[d] for d in donor])
    is_ref = core.make_pools(donor, sub_idx, nC)
    t_load = time.time() - t0

    train_ref = (split == "train") & is_ref
    R_rna_full, R_adt_full, n_full = core.build_ref(RNA, ADT, sub_idx, train_ref)
    gene_sel = core.select_features(R_rna_full)
    RNA_SEL = RNA[gene_sel].copy()
    CELL_RNA_TOT, CELL_ADT_TOT = RNA.sum(0), ADT.sum(0)

    REF = {("cross_full", None): dict(Rr=R_rna_full[gene_sel], Ra=R_adt_full, n=n_full,
                                      hash=core.ref_hash(R_rna_full, R_adt_full, gene_sel))}
    import zlib
    for d in SPLIT["val"] + SPLIT["test"]:
        Rr, Ra, n_sd = core.build_ref(RNA, ADT, sub_idx, (donor == d) & is_ref)
        REF[("same_donor", d)] = dict(Rr=Rr[gene_sel], Ra=Ra, n=n_sd,
                                      hash=core.ref_hash(Rr, Ra, gene_sel))
        rngm = np.random.default_rng(zlib.crc32(f"matched|{d}".encode()))
        mm = np.zeros(nC, bool)
        for j in range(len(SUB)):
            pool = np.where(train_ref & (sub_idx == j))[0]
            mm[rngm.choice(pool, min(n_sd[j], len(pool)), replace=False)] = True
        Rr2, Ra2, n_cm = core.build_ref(RNA, ADT, sub_idx, mm)
        REF[("cross_matched", d)] = dict(Rr=Rr2[gene_sel], Ra=Ra2, n=n_cm,
                                         hash=core.ref_hash(Rr2, Ra2, gene_sel))

    fo = open(f"{args.output_dir}/results_p2.csv", "w", newline="")
    W = csv.DictWriter(fo, ROWCOLS); W.writeheader()
    go = open(f"{args.output_dir}/failures_p2.csv", "w", newline="")
    FW = csv.DictWriter(go, FAILCOLS); FW.writeheader()
    den = collections.Counter(); nrow = 0

    for seed in CFG["seeds"]:
        for spl in ("val", "test"):
            genm = (split == spl) & (~is_ref)
            dh = sorted(set(donor[genm]))
            pools = {d: {j: np.where(genm & (donor == d) & (sub_idx == j))[0]
                         for j in range(len(SUB))} for d in dh}
            for scn in ("S0", "S1", "S2", "S3"):
                for (delta, eta) in core.grid_points(scn):
                    rng = core.spot_rng(seed, spl, scn, delta, eta)
                    pi_set, s_set = core.truth(delta, eta)
                    for k in range(CFG["n_spots"]):
                        d_k = dh[k % len(dh)]
                        for a_ in ARMS: den[(a_[0], a_[1], spl, "attempted")] += 1
                        got = core.make_spot(pools[d_k], pi_set, s_set, rng, RNA_SEL, ADT,
                                             sub_idx, CELL_RNA_TOT, CELL_ADT_TOT)
                        if got is None:
                            for dg, arm, rm, ia, st in ARMS:
                                den[(dg, arm, spl, "spot_failed")] += 1
                                FW.writerow(dict(diagnostic=dg, arm=arm, ref_mode=rm, info_arm=ia,
                                    scenario=scn, delta=delta, eta=eta, seed=seed, split=spl,
                                    donor=d_k, spot_id=k, reason="pool_exhausted"))
                            continue
                        (r, a), (pi_t, s_t), (m_r, m_a), ncell, cidx = got
                        chash = hashlib.sha256(",".join(map(str, sorted(cidx.tolist()))).encode()).hexdigest()[:16]
                        st1 = core.solve_free(REF[("cross_full", None)], r, a, "joint")
                        for dg, arm, rm, ia, st in ARMS:
                            refd = REF[(rm, None)] if rm == "cross_full" else REF[(rm, d_k)]
                            if dg == "D1":   out = core.solve_fixed_pi(refd, r, a, ia, pi_t)
                            elif dg == "D2": out = core.solve_free(refd, r, a, ia)
                            elif arm == "D3_stage1_joint": out = st1
                            else: out = None if st1 is None else core.solve_fixed_pi(refd, r, a, ia, st1["pi"])
                            if out is not None and out["resid"] > CFG["constraint_tol"]:
                                den[(dg, arm, spl, "constraint_violation")] += 1
                                FW.writerow(dict(diagnostic=dg, arm=arm, ref_mode=rm, info_arm=ia,
                                    scenario=scn, delta=delta, eta=eta, seed=seed, split=spl,
                                    donor=d_k, spot_id=k, reason="constraint_violation")); continue
                            if out is None:
                                den[(dg, arm, spl, "solver_failed")] += 1
                                FW.writerow(dict(diagnostic=dg, arm=arm, ref_mode=rm, info_arm=ia,
                                    scenario=scn, delta=delta, eta=eta, seed=seed, split=spl,
                                    donor=d_k, spot_id=k, reason="nnls_degenerate")); continue
                            den[(dg, arm, spl, "ok")] += 1
                            for ci, c in enumerate(CO):
                                W.writerow(dict(diagnostic=dg, arm=arm, ref_mode=rm, info_arm=ia,
                                    stage1_model=st, scenario=scn, delta=delta, eta=eta,
                                    depth_thin=CFG["depth_thin"], adt_dropout=CFG["adt_dropout"],
                                    seed=seed, split=spl, donor=d_k, spot_id=k, n_cells_true=ncell,
                                    cell_idx_sha256=chash, coarse_type=c,
                                    pi_true=f"{pi_t[ci]:.6f}", pi_hat=f"{out['pi'][ci]:.6f}",
                                    s_true=("" if np.isnan(s_t[ci]) else f"{s_t[ci]:.6f}"),
                                    s_hat=f"{out['s'][ci]:.6f}",
                                    theta_true=("" if np.isnan(s_t[ci]) else f"{pi_t[ci]*s_t[ci]:.6f}"),
                                    theta_hat=f"{out['pi'][ci]*out['s'][ci]:.6f}",
                                    m_rna=f"{m_r[ci]:.6f}", m_adt=f"{m_a[ci]:.6f}",
                                    constraint_resid=f"{out['resid']:.3e}",
                                    T_fitted=f"{out['T']:.6f}", status="ok"))
                                nrow += 1
            print(f"  seed={seed} split={spl} rows={nrow} t={time.time()-t0:.0f}s", flush=True)
    fo.close(); go.close()
    json.dump({f"{d}|{a}|{s}|{w}": v for (d, a, s, w), v in sorted(den.items())},
              open(f"{args.output_dir}/counts_p2.json", "w"), indent=1)
    man = dict(rows=nrow, runtime_compute_sec=round(time.time()-t0-t_load, 1),
               runtime_load_and_setup_sec=round(t_load, 1),
               runtime_total_in_process_sec=round(time.time()-t0, 1),
               threads=args.threads, numpy=np.__version__, pandas=pd.__version__,
               python=sys.version.split()[0], n_genes_selected=int(len(gene_sel)),
               reference_arms={f"{k[0]}|{k[1]}": {"per_subtype_cells": v["n"].tolist(),
                                                  "hash": v["hash"]} for k, v in REF.items()},
               cfg={k: v for k, v in CFG.items()},
               note="nominal arm only (depth_thin=1.0, adt_dropout=0.0)")
    json.dump(man, open(f"{args.output_dir}/manifest_p2.json", "w"), indent=1, default=str)
    print(f"DONE rows={nrow} compute={man['runtime_compute_sec']}s load={man['runtime_load_and_setup_sec']}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
