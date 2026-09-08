"""Shared math core.  Carried over VERBATIM from the frozen 2026-09-07 pilot and
the 2026-09-08 P2 diagnostics.  No algorithmic change -- only relocation."""
import hashlib, zlib
import numpy as np
from scipy.optimize import nnls
from .config import CFG, COARSE, CO, SUB, SUB_OF_C, MONO

def set_threads(n=4):
    """Must run BEFORE numpy is imported by the process to take effect."""
    import os
    for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(n)

# ---- pools -----------------------------------------------------------------
def make_pools(donor, sub_idx, n_cells, ref_pool_frac=None, ref_seed=None):
    """Disjoint reference / generation pools drawn within donor x subtype."""
    frac = CFG["ref_pool_frac"] if ref_pool_frac is None else ref_pool_frac
    seed = CFG["ref_seed"] if ref_seed is None else ref_seed
    rng = np.random.default_rng(seed)
    is_ref = np.zeros(n_cells, bool)
    for d in np.unique(donor):
        for j in range(len(SUB)):
            idx = np.where((donor == d) & (sub_idx == j))[0]
            if len(idx) == 0: continue
            is_ref[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = True
    return is_ref

def build_ref(RNA, ADT, sub_idx, mask):
    """RAW per-cell mean profiles.  IDENTICAL formula for every reference arm."""
    nG, nA = RNA.shape[0], ADT.shape[0]
    Rr = np.zeros((nG, len(SUB))); Ra = np.zeros((nA, len(SUB))); n = np.zeros(len(SUB), int)
    for j in range(len(SUB)):
        m = mask & (sub_idx == j); n[j] = int(m.sum())
        if n[j] == 0: return None, None, None
        Rr[:, j] = RNA[:, m].mean(1); Ra[:, j] = ADT[:, m].mean(1)
    return Rr, Ra, n

def select_features(R_rna, n_genes=None):
    """TRAIN-ONLY between-subtype variance ranking; frozen once, shared by all arms."""
    k = CFG["n_genes"] if n_genes is None else n_genes
    v = R_rna.var(1) / (R_rna.mean(1) + 1e-12)
    return np.argsort(-v)[:min(k, R_rna.shape[0])]

def ref_hash(Rr, Ra, gene_sel):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(Rr[gene_sel]).tobytes())
    h.update(np.ascontiguousarray(Ra).tobytes())
    return h.hexdigest()[:16]

# ---- scenarios and spot synthesis ------------------------------------------
def grid_points(scn):
    if scn == "S0": return [(0.0, 0.0)]
    if scn == "S1": return [(d, 0.0) for d in CFG["S1_delta"]]
    if scn == "S2": return [(0.0, e) for e in CFG["S2_eta"]]
    return [(d, e) for d in CFG["S3_delta"] for e in CFG["S3_eta"]]

def truth(delta, eta):
    pi = np.array(CFG["pi0"], float) + delta * np.array([1.0, -1.0, 0.0, 0.0])
    s = np.full(len(CO), CFG["s_S0"]); s[MONO] = CFG["s_S0"] + eta
    return pi, s

def spot_rng(seed, spl, scn, delta, eta, thin=None, drop=None):
    """Frozen rng key -- identical string to the 9/7 nominal arm."""
    thin = CFG["depth_thin"] if thin is None else thin
    drop = CFG["adt_dropout"] if drop is None else drop
    return np.random.default_rng(zlib.crc32(f"{seed}|{spl}|{scn}|{delta}|{eta}|{thin}|{drop}".encode()))

def make_spot(pool_by_sub, pi, s, rng, RNA_SEL, ADT, sub_idx, CELL_RNA_TOT, CELL_ADT_TOT):
    """Nominal arm: no depth thinning, no ADT dropout.  Cells come from ONE donor."""
    n = CFG["n_cells_per_spot"]; want = {}
    for ci, c in enumerate(CO):
        nc = int(round(pi[ci] * n)); a = int(round(s[ci] * nc))
        want[COARSE[c][0]] = a; want[COARSE[c][1]] = nc - a
    picks = []
    for sname, k in want.items():
        j = SUB.index(sname); avail = pool_by_sub[j]
        if k <= 0: continue
        if len(avail) < k: return None
        picks.append(rng.choice(avail, k, replace=False))
    if not picks: return None
    idx = np.concatenate(picks)
    r = RNA_SEL[:, idx].sum(1); a = ADT[:, idx].sum(1)
    cnt = np.bincount(sub_idx[idx], minlength=len(SUB)).astype(float)
    pi_t = np.array([cnt[SUB_OF_C[c]].sum() for c in CO]); pi_t /= pi_t.sum()
    s_t = np.array([(cnt[SUB_OF_C[c][0]] / cnt[SUB_OF_C[c]].sum())
                    if cnt[SUB_OF_C[c]].sum() >= 3 else np.nan for c in CO])
    mr = np.array([CELL_RNA_TOT[idx][np.isin(sub_idx[idx], SUB_OF_C[c])].sum() for c in CO])
    ma = np.array([CELL_ADT_TOT[idx][np.isin(sub_idx[idx], SUB_OF_C[c])].sum() for c in CO])
    return (r, a), (pi_t, s_t), (mr / mr.sum(), ma / max(ma.sum(), 1e-9)), len(idx), idx

# ---- solvers ---------------------------------------------------------------
def design(refd, info_arm):
    """Block construction and Frobenius-norm block weighting (9/7 convention)."""
    A = refd["Rr"]; wR = 1.0 / max(np.linalg.norm(A), 1e-12); blocks = [A * wR]
    wA = None
    if info_arm == "joint":
        Aa = refd["Ra"]; wA = CFG["lam_adt"] / max(np.linalg.norm(Aa), 1e-12)
        blocks.append(Aa * wA)
    return np.vstack(blocks), (wR, wA)

def rhs(r, a, w, info_arm):
    wR, wA = w
    return np.concatenate([r * wR, a * wA]) if info_arm == "joint" else r * wR

def _unpack(x):
    cl = x / x.sum()
    pi = np.array([cl[SUB_OF_C[c]].sum() for c in CO])
    s = np.array([cl[SUB_OF_C[c][0]] / max(cl[SUB_OF_C[c]].sum(), 1e-12) for c in CO])
    return pi, s

def solve_free(refd, r, a, info_arm):
    A, w = design(refd, info_arm); x, _ = nnls(A, rhs(r, a, w, info_arm))
    if x.sum() <= 0: return None
    pi, s = _unpack(x)
    return dict(pi=pi, s=s, T=float(x.sum()), resid=0.0)

def solve_fixed_pi(refd, r, a, info_arm, pi_given):
    """HOMOGENEOUS constraint  sum_{j in c} x_j - pi_c * sum_j x_j = 0.
    Fixes coarse PROPORTIONS only; the total is fitted freely from the data.
    The true cell count N is NEVER supplied."""
    A, w = design(refd, info_arm); y = rhs(r, a, w, info_arm)
    C = np.zeros((len(CO), len(SUB)))
    for ci, c in enumerate(CO):
        C[ci, :] = -pi_given[ci]; C[ci, SUB_OF_C[c]] += 1.0
    rho = CFG["rho_mult"] * max(np.linalg.norm(A), 1e-12)
    x, _ = nnls(np.vstack([A, rho * C]), np.concatenate([y, np.zeros(len(CO))]))
    if x.sum() <= 0: return None
    pi, s = _unpack(x)
    return dict(pi=pi, s=s, T=float(x.sum()), resid=float(np.abs(pi - pi_given).max()))
