import unittest

import numpy as np

from bridge_experiment.run_bridge import select_eval_subset


class BridgeExperimentTest(unittest.TestCase):
    def test_select_eval_subset_is_deterministic(self):
        features = np.arange(40).reshape(20, 2)
        labels = np.arange(20)

        first_x, first_y = select_eval_subset(features, labels, sample_size=5, seed=42)
        second_x, second_y = select_eval_subset(features, labels, sample_size=5, seed=42)

        np.testing.assert_array_equal(first_x, second_x)
        np.testing.assert_array_equal(first_y, second_y)
        self.assertEqual(len(first_y), 5)

    def test_select_eval_subset_returns_all_when_sample_is_zero(self):
        features = np.arange(12).reshape(6, 2)
        labels = np.arange(6)

        selected_x, selected_y = select_eval_subset(features, labels, sample_size=0, seed=42)

        np.testing.assert_array_equal(selected_x, features)
        np.testing.assert_array_equal(selected_y, labels)


if __name__ == "__main__":
    unittest.main()
