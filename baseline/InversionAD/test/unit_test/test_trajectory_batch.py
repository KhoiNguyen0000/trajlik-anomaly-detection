import unittest

import torch

from src.trajectory_batch import build_trajectory_batch


class TrajectoryBatchTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.states = torch.randn(2, 4, 8, 3, 3)
        self.epsilons = torch.randn(2, 3, 8, 3, 3)
        self.deltas = self.states[:, 1:] - self.states[:, :-1]

    def raw_online_output(self):
        return {
            "z_0": self.states[:, 0],
            "z_seq": list(self.states[:, 1:].unbind(dim=1)),
            "eps_seq": list(self.epsilons.unbind(dim=1)),
            "delta_z_seq": list(self.deltas.unbind(dim=1)),
        }

    def raw_cached_output(self):
        return {
            "z_0": self.states[:, 0],
            "z_seq": self.states[:, 1:],
            "eps_seq": self.epsilons,
            "delta_z_seq": self.deltas,
        }

    def test_online_and_cached_outputs_match(self):
        online = build_trajectory_batch(self.raw_online_output())
        cached = build_trajectory_batch(self.raw_cached_output())

        self.assertEqual(tuple(online["states"].shape), (2, 4, 8, 3, 3))
        self.assertEqual(tuple(online["epsilons"].shape), (2, 3, 8, 3, 3))
        self.assertEqual(tuple(online["deltas"].shape), (2, 3, 8, 3, 3))
        self.assertEqual(tuple(online["a_end_coarse"].shape), (2, 3, 3))

        for key in online:
            torch.testing.assert_close(online[key], cached[key])

    def test_canonical_input_is_supported(self):
        canonical = build_trajectory_batch(
            {
                "states": self.states,
                "epsilons": self.epsilons,
            }
        )

        torch.testing.assert_close(canonical["z0"], self.states[:, 0])
        torch.testing.assert_close(canonical["deltas"], self.deltas)

    def test_unbatched_cache_gets_batch_dimension(self):
        canonical = build_trajectory_batch(
            {
                "z_0": self.states[0, 0],
                "z_seq": self.states[0, 1:],
                "eps_seq": self.epsilons[0],
                "delta_z_seq": self.deltas[0],
            }
        )

        self.assertEqual(tuple(canonical["states"].shape), (1, 4, 8, 3, 3))

    def test_inconsistent_delta_is_rejected(self):
        output = self.raw_cached_output()
        output["delta_z_seq"] = output["delta_z_seq"] + 1.0

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            build_trajectory_batch(output)


if __name__ == "__main__":
    unittest.main()
