#!/usr/bin/env python3
"""Generate a tiny SYNTHETIC fixture with the same schema as the real inputs.
Contains no real data and is safe to commit."""
import os, numpy as np, pandas as pd
SUB = ["B naive","B memory","CD4 Naive","CD4 TCM","CD8 TEM","CD8 Naive","CD14 Mono","CD16 Mono"]
DONORS = [f"P{i}" for i in range(1, 9)]

def make(out_dir, n_per=60, n_genes=200, n_adt=20, seed=7):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = [(f"{d}_{s.replace(' ','')}_{i}", d, s)
            for d in DONORS for s in SUB for i in range(n_per)]
    cells = pd.DataFrame(rows, columns=["cell_id","donor","celltype.l2"])
    nC = len(cells)
    sidx = cells["celltype.l2"].map({s: i for i, s in enumerate(SUB)}).to_numpy()
    # subtype-specific expression programmes so the references are separable
    g_mu = rng.gamma(2.0, 1.0, size=(n_genes, len(SUB))) * rng.uniform(.5, 3, (1, len(SUB)))
    a_mu = rng.gamma(3.0, 2.0, size=(n_adt,  len(SUB))) * rng.uniform(.5, 3, (1, len(SUB)))
    RNA = rng.poisson(g_mu[:, sidx]).astype(np.int64)
    ADT = rng.poisson(a_mu[:, sidx]).astype(np.int64)
    def coo(M):
        r, c = np.nonzero(M); return r.astype(np.int64), c.astype(np.int64), M[r, c].astype(np.int64)
    rr, rc, rv = coo(RNA); ar, ac, av = coo(ADT)
    np.savez_compressed(os.path.join(out_dir, "pilot_counts.npz"),
        cells=cells.cell_id.to_numpy().astype("<U40"),
        rna_genes=np.array([f"G{i}" for i in range(n_genes)], dtype="<U16"),
        adt_features=np.array([f"A{i}" for i in range(n_adt)], dtype="<U13"),
        rna_row=rr, rna_col=rc, rna_val=rv, adt_row=ar, adt_col=ac, adt_val=av,
        rna_shape=np.array([n_genes, nC]), adt_shape=np.array([n_adt, nC]))
    cells.to_csv(os.path.join(out_dir, "pilot_cells.csv"), index=False)
    return out_dir

if __name__ == "__main__":
    import sys
    print(make(sys.argv[1] if len(sys.argv) > 1 else "./fixture"))
