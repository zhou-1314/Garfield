#!/usr/bin/env python3
"""Unit tests for the portable compstate package.  Synthetic data only."""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from compstate import core, io_data
from compstate.config import CFG, CO, SUB, SUB_OF_C, SPLIT, SPOTKEY, PRIVILEGED
from tests.make_fixture import make

_FIX = None
def fixture():
    global _FIX
    if _FIX is None:
        _FIX = make(os.path.join(tempfile.mkdtemp(prefix="compstate_fix_"), "data"))
    return _FIX

class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.RNA, cls.ADT, cls.cells = io_data.load(fixture())
        cls.nC = len(cls.cells)
        cls.sub_idx = cls.cells["celltype.l2"].map({s: i for i, s in enumerate(SUB)}).to_numpy()
        cls.donor = cls.cells["donor"].to_numpy()
        so = {d: k for k, v in SPLIT.items() for d in v}
        cls.split = np.array([so[d] for d in cls.donor])
        cls.is_ref = core.make_pools(cls.donor, cls.sub_idx, cls.nC)
        Rr, Ra, n = core.build_ref(cls.RNA, cls.ADT, cls.sub_idx, (cls.split == "train") & cls.is_ref)
        cls.gsel = core.select_features(Rr, n_genes=100)
        cls.refd = dict(Rr=Rr[cls.gsel], Ra=Ra, n=n)
        cls.RNA_SEL = cls.RNA[cls.gsel].copy()
        cls.CRT, cls.CAT = cls.RNA.sum(0), cls.ADT.sum(0)

    def a_spot(self, delta=0.15, eta=-0.3, spl="test"):
        genm = (self.split == spl) & (~self.is_ref)
        d0 = sorted(set(self.donor[genm]))[0]
        pools = {j: np.where(genm & (self.donor == d0) & (self.sub_idx == j))[0]
                 for j in range(len(SUB))}
        pi, s = core.truth(delta, eta)
        got = core.make_spot(pools, pi, s, np.random.default_rng(0), self.RNA_SEL, self.ADT,
                             self.sub_idx, self.CRT, self.CAT)
        self.assertIsNotNone(got, "fixture pools too small to build a spot")
        return got

class TestConstraint(Base):
    def test_fixed_pi_residual_within_tol(self):
        (r, a), (pi_t, _), _, _, _ = self.a_spot()
        for ia in ("rna", "joint"):
            out = core.solve_fixed_pi(self.refd, r, a, ia, pi_t)
            self.assertLess(out["resid"], CFG["constraint_tol"],
                            f"{ia}: constraint residual exceeds tol")
            np.testing.assert_allclose(out["pi"], pi_t, atol=1e-6)

    def test_scale_invariance_proves_N_not_used(self):
        """y -> alpha*y must leave pi,s unchanged; only the fitted total scales."""
        (r, a), (pi_t, _), _, _, _ = self.a_spot()
        al = 7.3
        for solve, kw in ((core.solve_fixed_pi, dict(pi_given=pi_t)), (core.solve_free, {})):
            o1 = solve(self.refd, r, a, "joint", **kw)
            o2 = solve(self.refd, r*al, a*al, "joint", **kw)
            np.testing.assert_allclose(o1["pi"], o2["pi"], atol=1e-9)
            np.testing.assert_allclose(o1["s"], o2["s"], atol=1e-9)
            self.assertAlmostEqual(o2["T"]/o1["T"], al, places=6)

    def test_constraint_rows_are_homogeneous(self):
        """Scaling x must not change the constraint value => proportions only, no scale."""
        pi = np.array([.25,.25,.25,.25]); C = np.zeros((len(CO), len(SUB)))
        for ci, c in enumerate(CO):
            C[ci, :] = -pi[ci]; C[ci, SUB_OF_C[c]] += 1.0
        x = np.random.default_rng(1).random(len(SUB))
        np.testing.assert_allclose(C @ (3.7*x), 3.7*(C @ x), atol=1e-12)

class TestTheta8(Base):
    def _rows(self, arm, out, pi_t, s_t):
        return pd.DataFrame([dict(arm=arm, diagnostic="X", split="test", donor="P4",
            scenario="S3", delta=0.15, eta=-0.3, seed=1, spot_id=0, coarse_type=c,
            pi_true=pi_t[i], pi_hat=out["pi"][i], s_true=s_t[i], s_hat=out["s"][i])
            for i, c in enumerate(CO)])

    def test_theta8_equals_theta4_only_when_pi_is_fixed(self):
        (r, a), (pi_t, s_t), _, _, _ = self.a_spot()
        fixed = core.solve_fixed_pi(self.refd, r, a, "joint", pi_t)
        free = core.solve_free(self.refd, r, a, "joint")
        for out, expect_equal in ((fixed, True), (free, False)):
            # pi is enforced by a penalty block, so equality holds only to the
            # constraint residual -- not exactly.  Tie the tolerance to it.
            tol = max(10*out["resid"], 1e-9)
            d = self._rows("a", out, pi_t, s_t)
            t1 = (d.pi_hat*d.s_hat - d.pi_true*d.s_true).abs()
            t2 = (d.pi_hat*(1-d.s_hat) - d.pi_true*(1-d.s_true)).abs()
            mae4, mae8 = t1.mean(), (t1.sum()+t2.sum())/(2*len(d))
            if expect_equal:
                self.assertLess(abs(mae4-mae8), tol,
                    "with pi fixed (to within the constraint residual) the two subtypes carry "
                    "equal error, so 4-dim and 8-dim must agree to that same tolerance")
            else:
                self.assertGreater(abs(mae4-mae8), 1e-6,
                    "with pi free the 4-dim figure is biased; 8-dim must differ")

class TestAggregation(Base):
    def test_spotkey_contains_grid_point(self):
        """Regression guard: omitting delta/eta collapses distinct grid points."""
        self.assertIn("delta", SPOTKEY); self.assertIn("eta", SPOTKEY)

    def test_grid_points_not_collapsed(self):
        n_spots = 3
        rec = []
        for scn in ("S0", "S1", "S2", "S3"):
            for (delta, eta) in core.grid_points(scn):
                for k in range(n_spots):
                    rec.append(dict(arm="a", diagnostic="D", split="test", donor="P4",
                                    scenario=scn, delta=delta, eta=eta, seed=1, spot_id=k))
        df = pd.DataFrame(rec)
        got = df.groupby(SPOTKEY).ngroups
        self.assertEqual(got, len(df), "each (grid point, spot_id) must stay a distinct spot")
        per = df.groupby("scenario").apply(lambda d: d.groupby(SPOTKEY).ngroups)
        for scn, ngrid in (("S0",1), ("S1",3), ("S2",3), ("S3",4)):
            self.assertEqual(per[scn], ngrid*n_spots, f"{scn} collapsed")

class TestReferences(Base):
    def test_reference_arms_use_identical_formula(self):
        d0 = SPLIT["test"][0]
        sd = core.build_ref(self.RNA, self.ADT, self.sub_idx, (self.donor == d0) & self.is_ref)
        self.assertIsNotNone(sd[0])
        manual = self.RNA[:, (self.donor == d0) & self.is_ref &
                          (self.sub_idx == 0)].mean(1)
        np.testing.assert_allclose(sd[0][:, 0], manual, atol=1e-12)

    def test_pools_disjoint_within_each_donor(self):
        for d in set(self.donor):
            ref = (self.donor == d) & self.is_ref
            gen = (self.donor == d) & (~self.is_ref)
            self.assertEqual(int((ref & gen).sum()), 0)

    def test_privileged_arms_declared(self):
        self.assertEqual(PRIVILEGED, {"D1_truepi_rna","D1_truepi_joint","D2_same_donor"})

class TestInputErrors(unittest.TestCase):
    def test_missing_raw_explains_schema_and_source(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(io_data.InputMissingError) as cm:
                io_data.load(d)
            m = str(cm.exception)
            for token in ("pilot_counts.npz", "pilot_cells.csv", "GSE164378",
                          "rna_row", "celltype.l2", "LICENCE"):
                self.assertIn(token, m, f"error message must mention {token}")

    def test_cell_order_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as d:
            src = fixture()
            import shutil
            shutil.copy(os.path.join(src, "pilot_counts.npz"), d)
            c = pd.read_csv(os.path.join(src, "pilot_cells.csv"))
            c.loc[0, "cell_id"] = "TAMPERED"
            c.to_csv(os.path.join(d, "pilot_cells.csv"), index=False)
            with self.assertRaises(io_data.SchemaError):
                io_data.load(d)

if __name__ == "__main__":
    unittest.main(verbosity=2)
