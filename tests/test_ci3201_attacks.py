import unittest

import numpy as np

from ci3201_adversarial.attacks import (
    constrained_numeric_pgd_attack,
    fgsm_attack,
    numeric_feature_mask,
    pgd_attack,
)


class AttackSmokeTest(unittest.TestCase):
    def setUp(self):
        import torch
        from torch import nn

        self.model = nn.Sequential(nn.Linear(3, 1))
        with torch.no_grad():
            self.model[0].weight[:] = torch.tensor([[1.0, -1.0, 0.5]])
            self.model[0].bias[:] = torch.tensor([0.0])

        self.features = np.asarray([[0.2, 0.1, 1.0], [0.4, 0.8, 0.0]], dtype="float32")
        self.labels = np.asarray([1, 0], dtype=int)

    def test_fgsm_changes_features_within_expected_step(self):
        adversarial = fgsm_attack(self.model, self.features, self.labels, epsilon=0.1)
        max_change = np.max(np.abs(adversarial - self.features))

        self.assertEqual(adversarial.shape, self.features.shape)
        self.assertLessEqual(max_change, 0.100001)
        self.assertGreater(max_change, 0.0)

    def test_pgd_respects_epsilon_budget(self):
        adversarial = pgd_attack(
            self.model,
            self.features,
            self.labels,
            epsilon=0.2,
            step_size=0.05,
            steps=5,
        )

        self.assertLessEqual(np.max(np.abs(adversarial - self.features)), 0.200001)

    def test_constrained_attack_changes_only_numeric_mask(self):
        mask = np.asarray([True, False, True])
        adversarial = constrained_numeric_pgd_attack(
            self.model,
            self.features,
            self.labels,
            numeric_mask=mask,
            epsilon=0.2,
            step_size=0.05,
            steps=5,
        )

        np.testing.assert_allclose(adversarial[:, 1], self.features[:, 1])
        self.assertGreater(np.max(np.abs(adversarial[:, [0, 2]] - self.features[:, [0, 2]])), 0.0)

    def test_numeric_feature_mask_uses_preprocessor_prefix(self):
        names = ["num__dur", "cat__proto_tcp", "num__sbytes"]
        np.testing.assert_array_equal(numeric_feature_mask(names), np.asarray([True, False, True]))


if __name__ == "__main__":
    unittest.main()
