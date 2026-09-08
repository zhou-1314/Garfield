"""Frozen constants.  Values identical to the 2026-09-08 P2 protocol."""
SPLIT = {"train": ["P1","P2","P5","P6"], "val": ["P3","P7"], "test": ["P4","P8"]}
COARSE = {"Mono": ["CD14 Mono","CD16 Mono"], "CD4 T": ["CD4 Naive","CD4 TCM"],
          "CD8 T": ["CD8 TEM","CD8 Naive"], "B": ["B naive","B memory"]}
CO  = sorted(COARSE)
SUB = [s for c in CO for s in COARSE[c]]
SUB_OF_C = {c: [SUB.index(s) for s in COARSE[c]] for c in CO}
MONO = CO.index("Mono")

CFG = dict(
    ref_pool_frac=0.5, ref_seed=20260907, n_cells_per_spot=50,
    n_spots=40, seeds=[1,2,3], n_genes=1500,
    depth_thin=1.0, adt_dropout=0.0,          # nominal arm only
    s_S0=0.5, pi0=[0.25,0.25,0.25,0.25],
    S1_delta=[-.15,0.0,.15], S2_eta=[-.3,0.0,.3],
    S3_delta=[-.15,.15], S3_eta=[-.3,.3],
    lam_adt=1.0, rho_mult=1e3, constraint_tol=1e-6,
    norm_convention="np.linalg.norm on a 2-D array = FROBENIUS norm",
    solver=("scipy.optimize.nnls on RAW-scale blocks. References are per-subtype MEAN PER-CELL RAW "
            "counts; the response is the RAW pseudo-spot count vector. For D1/D3 a homogeneous "
            "penalty block enforces coarse proportions only; the total is fitted freely."),
    coefficient_interpretation=(
        "NNLS coefficients admit a cell-count-scale reading ONLY UNDER the working assumption that "
        "the reference scale matches the measurement conditions of the assayed sample. That "
        "assumption is UNPROVEN -- capture efficiency, background and composition may still differ "
        "across donors and reference pools, and whether it holds is part of what D2 diagnoses."),
)

# (diagnostic, arm, ref_mode, info_arm, stage1_model)
ARMS = [("D1","D1_truepi_rna",    "cross_full",   "rna",  ""),
        ("D1","D1_truepi_joint",  "cross_full",   "joint",""),
        ("D2","D2_cross_full",    "cross_full",   "joint",""),
        ("D2","D2_same_donor",    "same_donor",   "joint",""),
        ("D2","D2_cross_matched", "cross_matched","joint",""),
        ("D3","D3_stage1_joint",  "cross_full",   "joint","M4"),
        ("D3","D3_stage2_rna",    "cross_full",   "rna",  "M4")]

# arms that receive privileged information (NOT deployment performance, NOT an upper bound)
PRIVILEGED = {"D1_truepi_rna","D1_truepi_joint","D2_same_donor"}

ROWCOLS = ["diagnostic","arm","ref_mode","info_arm","stage1_model","scenario","delta","eta",
           "depth_thin","adt_dropout","seed","split","donor","spot_id","n_cells_true",
           "cell_idx_sha256","coarse_type","pi_true","pi_hat","s_true","s_hat",
           "theta_true","theta_hat","m_rna","m_adt","constraint_resid","T_fitted","status"]
FAILCOLS = ["diagnostic","arm","ref_mode","info_arm","scenario","delta","eta","seed",
            "split","donor","spot_id","reason"]
# spot identity MUST include the grid point: spot_id only runs 0..n_spots-1 WITHIN one (delta,eta).
SPOTKEY = ["arm","diagnostic","split","donor","scenario","delta","eta","seed","spot_id"]
