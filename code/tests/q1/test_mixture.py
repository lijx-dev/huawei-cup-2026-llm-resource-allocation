"""P3/M3 人工构造微型样例；绝不用于正式结果。"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import joblib

from q1.mixture.dataset import PAIRS, fields, load_pair
from q1.mixture.models import predict
from q1.mixture.perturbation import single, pair, check_simplex, interaction_value
from q1.integration.mapping import build_mapping, project


class MixtureTests(unittest.TestCase):
    def test_split_roles_and_actual_dimensions(self):
        root = Path(__file__).resolve().parents[2]
        mixture_fields, loss_fields = fields(root)
        self.assertEqual((len(mixture_fields), len(loss_fields)), (17, 13))
        self.assertEqual([name for name, (_, _, role) in PAIRS.items() if role == "training"], ["train_1m"])
        self.assertEqual([name for name, (_, _, role) in PAIRS.items() if role == "estimated_reference"], ["est_10b", "est_70b"])

    def test_join_by_index_not_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "data/real_attachments/A_data_value/regmix_tables"
            base.mkdir(parents=True)
            (base / "train_mixture_1m.csv").write_text("index,train_the_pile_a,train_the_pile_b\n1,0.2,0.8\n2,0.7,0.3\n")
            (base / "train_pile_loss_1m.csv").write_text("index,metric/the_pile_a_val_loss\n2,5\n1,7\n")
            ids, x, y, audit = load_pair(tmp, "train_1m", ["train_the_pile_a", "train_the_pile_b"],
                                         ["metric/the_pile_a_val_loss"], 0.001)
            self.assertEqual(ids.tolist(), ["1", "2"])
            self.assertEqual(y[:, 0].tolist(), [7, 5])
            self.assertEqual(audit["invalid"], 0)

    def test_simplex_and_linear_interaction(self):
        x = np.array([[0.2, 0.3, 0.5], [0.4, 0.2, 0.4]])
        check_simplex(x)
        changed, ok = single(x, 0, 0.05)
        self.assertTrue(ok.all())
        check_simplex(changed)
        self.assertTrue(np.allclose(changed[:, 1] / changed[:, 2], x[:, 1] / x[:, 2]))
        j, _ = pair(x, 0, 1, 0.03, 0)
        k, _ = pair(x, 0, 1, 0, 0.03)
        both, _ = pair(x, 0, 1, 0.03, 0.03)
        check_simplex(both)
        coef = np.array([1., 2., 4.])
        self.assertTrue(np.allclose(interaction_value(x @ coef, j @ coef, k @ coef, both @ coef), 0, atol=1e-12))
        _, impossible = pair(x, 0, 1, 0.9, 0.9)
        self.assertFalse(impossible.any())

    def test_mapping_and_incomplete_coverage(self):
        import pandas as pd
        guide = pd.DataFrame([{"mixture_domain": "a", "quality_domain": "q1", "mapping_type": "direct"},
                              {"mixture_domain": "b", "quality_domain": "(none)", "mapping_type": "inferred"}])
        distance = np.array([[0., 1.], [1., 0.]])
        mm, _, _ = build_mapping(guide, ["a", "b"], ["q1", "q2"], distance, assisted=False)
        q17, qmix, coverage, complete = project(np.array([[0.4, 0.6], [1., 0.]]), mm, np.array([0.8, 0.2]))
        self.assertTrue(np.isnan(q17[1]))
        self.assertTrue(np.isnan(qmix[0]))
        self.assertAlmostEqual(qmix[1], 0.8)
        self.assertTrue(np.allclose(coverage, [0.4, 1.]))
        self.assertEqual(complete.tolist(), [False, True])
        full, _, _ = build_mapping(guide, ["a", "b"], ["q1", "q2"], distance, assisted=True)
        self.assertTrue(np.allclose(full.sum(axis=1), 1))
        x = np.array([[0.4, 0.6]])
        q = np.array([0.8, 0.2])
        _, qm, _, _ = project(x, full, q)
        self.assertAlmostEqual(qm[0], float((x @ full @ q)[0]))
        self.assertAlmostEqual(float((x @ full).sum()), 1.0)

    def test_model_save_and_data_role(self):
        from sklearn.linear_model import Ridge
        x = np.array([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]])
        y = np.array([[1., 2.], [2., 3.], [3., 4.]])
        fitted = {"ridge": [Ridge().fit(x[:, :1], y[:, j]) for j in range(2)]}
        before = predict(fitted, "ridge", x, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(fitted, path)
            after = predict(joblib.load(path), "ridge", x, 1)
        np.testing.assert_allclose(before, after)


if __name__ == "__main__":
    unittest.main()
