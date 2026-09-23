import unittest
from collections import OrderedDict

import torch

from ci3203_federated.privacy import (
    aggregate_private_updates,
    clip_update,
    state_difference,
    update_l2_norm,
)


class PrivacyMechanismTest(unittest.TestCase):
    def test_state_difference_uses_local_minus_global(self):
        global_state = OrderedDict(weight=torch.tensor([1.0, 2.0]))
        local_state = OrderedDict(weight=torch.tensor([2.5, 1.0]))

        update = state_difference(local_state, global_state)

        torch.testing.assert_close(update["weight"], torch.tensor([1.5, -1.0]))

    def test_clip_update_limits_l2_norm(self):
        update = OrderedDict(weight=torch.tensor([3.0, 4.0]))

        clipped = clip_update(update, max_norm=1.0)

        self.assertLessEqual(update_l2_norm(clipped), 1.000001)

    def test_zero_noise_private_aggregation_matches_weighted_update(self):
        global_state = OrderedDict(weight=torch.tensor([1.0]))
        local_states = [
            OrderedDict(weight=torch.tensor([2.0])),
            OrderedDict(weight=torch.tensor([5.0])),
        ]

        private_state = aggregate_private_updates(
            global_state=global_state,
            local_states=local_states,
            client_sizes=[1, 3],
            clip_norm=10.0,
            noise_multiplier=0.0,
        )

        torch.testing.assert_close(private_state["weight"], torch.tensor([4.25]))

    def test_non_floating_state_uses_client_state(self):
        global_state = OrderedDict(counter=torch.tensor(0, dtype=torch.long))
        local_states = [
            OrderedDict(counter=torch.tensor(3, dtype=torch.long)),
            OrderedDict(counter=torch.tensor(7, dtype=torch.long)),
        ]

        private_state = aggregate_private_updates(
            global_state=global_state,
            local_states=local_states,
            client_sizes=[1, 1],
            clip_norm=1.0,
            noise_multiplier=0.0,
        )

        torch.testing.assert_close(private_state["counter"], torch.tensor(3, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
