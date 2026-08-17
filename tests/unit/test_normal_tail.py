import math
import unittest

import torch

from trajlik.normal_tail import (
    EmpiricalTailCalibrator,
    normal_kfold_indices,
    normal_train_calibration_split,
)


class EmpiricalTailCalibratorTest(unittest.TestCase):
    def test_empirical_cdf_uses_smoothed_rank_formula(self):
        calibrator = EmpiricalTailCalibrator().fit(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([10.0, 20.0, 30.0]),
        )
        values = torch.tensor([0.0, 1.0, 1.5, 3.0, 4.0])

        tail = calibrator.endpoint_tail(values)

        counts = torch.tensor([0.0, 1.0, 1.0, 3.0, 3.0])
        expected_cdf = (1.0 + counts) / 5.0
        expected = -torch.log(1.0 - expected_cdf + 1e-6)
        torch.testing.assert_close(tail, expected, atol=2e-6, rtol=2e-6)
        self.assertAlmostEqual(tail[-1].item(), math.log(5.0), places=5)

    def test_fusion_upsampling_and_range_score(self):
        calibrator = EmpiricalTailCalibrator().fit(
            torch.arange(8.0),
            torch.arange(8.0),
        )
        endpoint = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
        path = torch.tensor([[4.0, 5.0, 6.0, 7.0]])

        output = calibrator(endpoint, path, output_size=(8, 8))

        self.assertEqual(tuple(output["coarse_map"].shape), (1, 2, 2))
        self.assertEqual(tuple(output["pixel_map"].shape), (1, 8, 8))
        expected_range = output["coarse_map"].max() - output["coarse_map"].min()
        torch.testing.assert_close(output["image_score"][0], expected_range)

    def test_calibrator_round_trip_state(self):
        calibrator = EmpiricalTailCalibrator().fit(
            torch.randn(10),
            torch.randn(12),
        )
        restored = EmpiricalTailCalibrator.from_state_dict(calibrator.state_dict())

        query = torch.randn(4)
        torch.testing.assert_close(
            calibrator.endpoint_tail(query),
            restored.endpoint_tail(query),
        )

    def test_normal_only_splits_are_disjoint_and_deterministic(self):
        training, calibration = normal_train_calibration_split(100)
        training_again, calibration_again = normal_train_calibration_split(100)

        self.assertEqual(training.numel(), 95)
        self.assertEqual(calibration.numel(), 5)
        self.assertEqual(set(training.tolist()) & set(calibration.tolist()), set())
        torch.testing.assert_close(training, training_again)
        torch.testing.assert_close(calibration, calibration_again)

        folds = normal_kfold_indices(10, num_folds=5)
        self.assertEqual(len(folds), 5)
        self.assertTrue(all(cal.numel() == 2 for _, cal in folds))


if __name__ == "__main__":
    unittest.main()
