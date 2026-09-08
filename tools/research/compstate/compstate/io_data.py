"""Input loading with explicit schema validation and actionable errors.

This package NEVER downloads anything.  If the raw inputs are absent the error
message states the expected schema and how the files are produced.
"""
import os
import numpy as np, pandas as pd
from .config import SUB

NPZ_NAME, CSV_NAME = "pilot_counts.npz", "pilot_cells.csv"
REQUIRED_NPZ = ["cells","rna_genes","adt_features","rna_row","rna_col","rna_val",
                "adt_row","adt_col","adt_val","rna_shape","adt_shape"]
REQUIRED_CSV = ["cell_id","donor","celltype.l2"]

SCHEMA_DOC = f"""
Expected inputs in --data-dir:

  {NPZ_NAME}   (COO triplets; ~93 MB for the full pilot subset)
    cells         <U    (n_cells,)      cell barcodes, ORDER MUST MATCH the CSV
    rna_genes     <U    (n_genes,)
    adt_features  <U    (n_adt,)
    rna_row/rna_col/rna_val   int64 (nnz_rna,)   densify: M[row,col]=val, rows=features
    adt_row/adt_col/adt_val   int64 (nnz_adt,)
    rna_shape/adt_shape       int64 (2,)

  {CSV_NAME}   (>=3 columns are USED; extra audit columns are allowed)
    cell_id      must equal npz['cells'] element-wise and in the same order
    donor        expected P1..P8
    celltype.l2  must cover the 8 subtypes: {SUB}

HOW TO PRODUCE THEM -- these files are an EXTERNAL INPUT PREMISE.
This package NEVER downloads anything and does not bundle the raw matrices.

  Official source: NCBI GEO accession GSE164378 (Hao et al., Cell 2021),
                   https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378
  Files needed   : the 3P (3-prime) arm only --
                     RNA_3P  raw count matrix + barcodes.tsv + features.tsv
                     ADT_3P  raw count matrix + barcodes.tsv + features.tsv
                     sc.meta.data_3P  (supplies donor and celltype.l2)
                   The RNA_3P and ADT_3P barcode files are byte-identical, i.e. both
                   modalities are measured on the SAME cells in the SAME order.
  Construction   : keep only the 8 subtypes listed above; within each
                   (donor, subtype) stratum draw up to a fixed cap of cells uniformly
                   at random with a fixed seed; compute raw library sums over the FULL
                   feature set BEFORE any gene subsetting; store both matrices as the
                   COO triplets described above with a shared cell order.
                   A reference implementation (export_pilot_subset.py) exists alongside
                   the original pilot data; it is NOT required -- any exporter producing
                   the schema above will work.
  LICENCE        : GSE164378 redistribution terms are UNVERIFIED and no licence file
                   accompanies the data. Treat redistribution as NOT established.
"""

class InputMissingError(FileNotFoundError):
    pass
class SchemaError(ValueError):
    pass

def load(data_dir):
    npz_p, csv_p = os.path.join(data_dir, NPZ_NAME), os.path.join(data_dir, CSV_NAME)
    missing = [p for p in (npz_p, csv_p) if not os.path.exists(p)]
    if missing:
        raise InputMissingError(
            "Missing required input file(s):\n  " + "\n  ".join(missing) + "\n" + SCHEMA_DOC)
    z = np.load(npz_p, allow_pickle=True)
    absent = [k for k in REQUIRED_NPZ if k not in z.files]
    if absent:
        raise SchemaError(f"{NPZ_NAME} is missing array(s) {absent}.\n{SCHEMA_DOC}")
    cells = pd.read_csv(csv_p)
    absent = [c for c in REQUIRED_CSV if c not in cells.columns]
    if absent:
        raise SchemaError(f"{CSV_NAME} is missing column(s) {absent}.\n{SCHEMA_DOC}")
    if list(z["cells"]) != list(cells["cell_id"]):
        raise SchemaError(
            "Cell order mismatch: npz['cells'] != csv['cell_id'].  The two files must list the "
            "same barcodes in the same order.\n" + SCHEMA_DOC)
    have = set(cells["celltype.l2"].unique())
    if not set(SUB).issubset(have):
        raise SchemaError(f"celltype.l2 is missing subtype(s) {sorted(set(SUB)-have)}.\n{SCHEMA_DOC}")
    nG, nA = int(z["rna_shape"][0]), int(z["adt_shape"][0])
    nC = len(cells)
    RNA = np.zeros((nG, nC)); RNA[z["rna_row"], z["rna_col"]] = z["rna_val"]
    ADT = np.zeros((nA, nC)); ADT[z["adt_row"], z["adt_col"]] = z["adt_val"]
    for name, v in (("rna_val", z["rna_val"]), ("adt_val", z["adt_val"])):
        if (v < 0).any():
            raise SchemaError(f"{name} contains negative counts; raw counts must be non-negative.")
    return RNA, ADT, cells
