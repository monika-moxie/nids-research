import unittest

import numpy as np

from ci3201_adversarial.smoothing import (
    Certificate,
    certified_accuracy_by_radius,
    certified_radius_from_probability,
    hoeffding_lower_bound,
)


class SmoothingMathTest(unittest.TestCase):
    def test_hoeffding_lower_bound_is_conservative(self):
        lower = hoeffding_lower_bound(successes=90, trials=100, alpha=0.01)

        self.assertLess(lower, 0.90)
        self.assertGreater(lower, 0.70)

    def test_radius_is_zero_without_majority_confidence(self):
        self.assertEqual(certified_radius_from_probability(0.50, sigma=0.25), 0.0)
        self.assertEqual(certified_radius_from_probability(0.40, sigma=0.25), 0.0)

    def test_radius_grows_with_probability(self):
        small = certified_radius_from_probability(0.70, sigma=0.25)
        large = certified_radius_from_probability(0.90, sigma=0.25)

        self.assertGreater(small, 0.0)
        self.assertGreater(large, small)

    def test_certified_accuracy_counts_correct_and_radius(self):
        certificates = [
            Certificate(1, 0.95, 0.90, 0.30, False),
            Certificate(0, 0.90, 0.80, 0.10, False),
            Certificate(1, 0.55, 0.45, 0.00, True),
        ]
        labels = np.asarray([1, 1, 1])

        result = certified_accuracy_by_radius(certificates, labels, [0.0, 0.2])

        self.assertEqual(result["0.0"], 1 / 3)
        self.assertEqual(result["0.2"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
