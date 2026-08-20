import unittest

import pandas as pd

from shared.config import DatasetConfig
from shared.data import prepare_data


class SharedPreprocessingTest(unittest.TestCase):
    def test_preprocessing_drops_leakage_and_handles_unknown_categories(self):
        train = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "dur": [0.1, 0.2, None, 0.4],
                "proto": ["tcp", "udp", "tcp", "icmp"],
                "service": ["http", "-", "dns", "http"],
                "attack_cat": ["Normal", "Generic", "Normal", "DoS"],
                "label": [0, 1, 0, 1],
            }
        )
        test = pd.DataFrame(
            {
                "id": [5, 6],
                "dur": [0.3, None],
                "proto": ["tcp", "sctp"],
                "service": ["http", "ftp"],
                "attack_cat": ["Normal", "Exploits"],
                "label": [0, 1],
            }
        )

        prepared = prepare_data(train, test, DatasetConfig())

        self.assertEqual(prepared.x_train.shape[0], 4)
        self.assertEqual(prepared.x_test.shape[0], 2)
        self.assertNotIn("id", " ".join(prepared.feature_names))
        self.assertNotIn("attack_cat", " ".join(prepared.feature_names))
        self.assertFalse(pd.isna(prepared.x_train).any())
        self.assertFalse(pd.isna(prepared.x_test).any())


if __name__ == "__main__":
    unittest.main()
