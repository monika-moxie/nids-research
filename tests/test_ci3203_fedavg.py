import unittest
from collections import OrderedDict

import numpy as np

from ci3203_federated.fedavg import (
    average_state_dicts,
    label_distribution,
    make_iid_partitions,
    make_label_skew_partitions,
)


class FedAvgTest(unittest.TestCase):
    def test_iid_partitions_cover_each_index_once(self):
        shards = make_iid_partitions(num_examples=10, num_clients=3, seed=7)
        combined = np.concatenate([shard.indices for shard in shards])

        self.assertEqual(sorted(combined.tolist()), list(range(10)))
        self.assertEqual(len(shards), 3)

    def test_label_skew_partitions_are_non_iid(self):
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        shards = make_label_skew_partitions(labels, num_clients=2, seed=7)

        first = label_distribution(labels, shards[0])
        second = label_distribution(labels, shards[1])

        self.assertGreater(first["normal"], first["attack"])
        self.assertGreater(second["attack"], second["normal"])

    def test_average_state_dicts_uses_client_sizes(self):
        states = [
            OrderedDict(weight=np.asarray([1.0, 3.0])),
            OrderedDict(weight=np.asarray([5.0, 7.0])),
        ]

        averaged = average_state_dicts(states, [1, 3])

        np.testing.assert_allclose(averaged["weight"], np.asarray([4.0, 6.0]))


if __name__ == "__main__":
    unittest.main()
