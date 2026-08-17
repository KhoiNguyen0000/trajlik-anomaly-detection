import sys
import unittest
from pathlib import Path

import numpy as np

baseline_root = str(Path(__file__).resolve().parents[2] / "baseline" / "InversionAD")
if baseline_root not in sys.path:
    sys.path.insert(0, baseline_root)

from src.adeval.eval_utils import calculate_img_metrics, calculate_px_metrics


class MetricDeviceTest(unittest.TestCase):
    def test_official_metrics_run_on_cpu(self):
        labels = np.array([0, 0, 1, 1], dtype=np.uint8)
        image_scores = np.array([0.0, 0.1, 0.8, 1.0], dtype=np.float32)
        masks = np.zeros((4, 8, 8), dtype=np.uint8)
        masks[2:, 2:6, 2:6] = 1
        pixel_scores = masks.astype(np.float32) + 0.01

        image_metrics = calculate_img_metrics(
            labels,
            image_scores,
            ["img_auroc", "img_f1max", "img_ap"],
            device="cpu",
        )
        pixel_metrics = calculate_px_metrics(
            masks,
            pixel_scores,
            ["px_auroc", "px_f1max", "px_ap", "px_aupro"],
            device="cpu",
        )

        self.assertTrue(all(np.isfinite(value) for value in image_metrics.values()))
        self.assertTrue(all(np.isfinite(value) for value in pixel_metrics.values()))


if __name__ == "__main__":
    unittest.main()
