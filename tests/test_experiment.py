"""Analytic metrics and a tiny synthetic integration test, not empirical evidence."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from delivery_risk.experiment import (ExperimentConfig,apply_policy,budget_policy,metric_row,preflight,paired_bootstrap,run_experiment)


class MetricsTests(unittest.TestCase):
    def test_ties_have_random_expected_recall(self):
        y=np.array([0,1]*50); s=np.ones(100)
        m=metric_row(y,s,False)
        self.assertAlmostEqual(m["ap"],.5)
        self.assertAlmostEqual(m["recall_budget_10pct"],.1)
        self.assertAlmostEqual(m["recall_lift_vs_random_budget_10pct"],1.)
    def test_fractional_boundary(self):
        s=np.array([5.,4.,4.,4.,0.]); p=budget_policy(s,.4)
        w=apply_policy(s,p)
        np.testing.assert_allclose(w,[1,1/3,1/3,1/3,0])
        self.assertAlmostEqual(w.sum(),2)
    def test_bootstrap_identical_models(self):
        y=np.array([0,1]*25); s=np.arange(50)/50
        b=paired_bootstrap(y,s,s,np.array([str(i//10) for i in range(50)]),ExperimentConfig(bootstrap_draws=10))
        self.assertEqual(b["ap_difference"]["ci95_low"],0)
        self.assertEqual(b["recall_at_10pct_difference"]["ci95_high"],0)
    def test_overlap_is_rejected(self):
        rows=[]
        for split in ("train","dev","eval"):
            row=dict(split=split,scenario_id=split,source_file=split,source_sha256="a"*64,ego_speed=1,cv_min_clearance=4)
            for t in (1,2,3): row.update({f"eligible_{t}m":1,f"label_{t}m":0})
            rows.append(row)
        with self.assertRaisesRegex(ValueError,"same_source_hash"):
            preflight(pd.DataFrame(rows),ExperimentConfig())


class IntegrationTests(unittest.TestCase):
    def test_tiny_fixture_remains_exploratory(self):
        # Deliberately tiny fixture; assert that completed execution is never promoted
        # into a sufficient empirical finding. Everything lives in a temporary dir.
        rng=np.random.default_rng(2026); rows=[]
        for i,split in enumerate(("train","dev","eval")):
            for j in range(16):
                row=dict(split=split,scenario_id=f"unit_fixture_{split}_{j}",source_file=f"fixture_{split}",source_sha256=str(i+1)*64,ego_speed=float(rng.uniform(1,10)),cv_min_clearance=float(rng.uniform(0,10)),neighbor_count=2.,n0_vx=float(rng.normal()),map_crosswalk_distance=10.)
                for t in (1,2,3): row.update({f"eligible_{t}m":1,f"label_{t}m":j%2})
                rows.append(row)
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"fixture.csv"; pd.DataFrame(rows).to_csv(source,index=False)
            result=run_experiment(source,Path(d)/"results",ExperimentConfig(bootstrap_draws=5))
            self.assertEqual(result["primary_status"],"exploratory_below_minimum_counts")
            self.assertTrue((Path(d)/"results"/"report.md").exists())


if __name__=="__main__": unittest.main()
