"""人工构造的配比优化微型样例；不参与正式候选搜索。"""
import unittest
import numpy as np

from q1.mixture.optimize import _weights, support_mask, transfer


class OptimizeTests(unittest.TestCase):
    def test_weight_validation(self):
        np.testing.assert_allclose(_weights(None, 3), [1 / 3] * 3)
        np.testing.assert_allclose(_weights([2, 1, 1], 3), [.5, .25, .25])
        for bad in ([0, 0, 0], [-1, 2, 1], [1, 2]):
            with self.assertRaises(ValueError):
                _weights(bad, 3)

    def test_transfer_preserves_simplex_and_rejects_shortfall(self):
        p = np.array([.2, .3, .5])
        changed = transfer(p, 0, 2, .05)
        np.testing.assert_allclose(changed, [.25, .3, .45])
        self.assertAlmostEqual(changed.sum(), 1)
        with self.assertRaises(ValueError):
            transfer(p, 0, 1, .4)

    def test_support_uses_distance_and_observed_share_caps(self):
        train = np.array([[.8, .2, 0.], [.2, .8, 0.], [.6, .1, .3]])
        candidate = np.array([[.75, .25, 0.], [.15, .85, 0.], [.4, .3, .3]])
        good, distance = support_mask(candidate, train, threshold=.2, batch_size=2)
        self.assertEqual(good.tolist(), [True, False, False])
        self.assertEqual(len(distance), 3)
        with self.assertRaises(ValueError):
            support_mask(np.array([[.5, .5, .5]]), train, threshold=.3)


if __name__ == "__main__":
    unittest.main()
