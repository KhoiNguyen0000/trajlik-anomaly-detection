import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import _bootstrap  # noqa: F401

from src.evaluate import (
    concat_all_gather,
    inversion_endpoint_outputs,
    resolve_evaluation_checkpoint,
)


class DenoiserStub:
    def ddim_reverse_sample(self, features, start_t, labels, eta):
        self.start_t = start_t
        return features + 1.0


class EvaluateContractTest(unittest.TestCase):
    def test_endpoint_outputs_are_batch_safe_and_channel_correct(self):
        denoiser = DenoiserStub()
        features = torch.randn(3, 2, 2, 3)
        labels = torch.zeros(3, dtype=torch.long)

        latent, endpoint_map, image_range, nll = inversion_endpoint_outputs(
            denoiser,
            features,
            labels,
            (5, 7),
        )

        self.assertEqual(tuple(denoiser.start_t.shape), (3,))
        self.assertEqual(tuple(endpoint_map.shape), (3, 5, 7))
        self.assertEqual(tuple(image_range.shape), (3,))
        self.assertEqual(tuple(nll.shape), (3,))
        expected_coarse = torch.linalg.vector_norm(latent.float(), dim=1)
        expected_range = expected_coarse.amax(dim=(-2, -1)) - expected_coarse.amin(
            dim=(-2, -1)
        )
        torch.testing.assert_close(image_range, expected_range)

    def test_single_process_gather_does_not_require_process_group(self):
        values = torch.tensor([1.0, 2.0])

        gathered = concat_all_gather(values, world_size=1)

        np.testing.assert_array_equal(gathered, np.array([1.0, 2.0]))

    def test_explicit_downloaded_checkpoint_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            checkpoint.touch()
            args = SimpleNamespace(
                checkpoint_path=str(checkpoint),
                save_dir=None,
                use_ema_model=False,
                use_best_model=False,
            )

            self.assertEqual(resolve_evaluation_checkpoint(args), str(checkpoint))


if __name__ == "__main__":
    unittest.main()
