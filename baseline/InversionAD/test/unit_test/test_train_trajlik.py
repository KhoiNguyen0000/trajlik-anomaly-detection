import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from src.train_trajlik import load_trajlik_checkpoint, train


class TrainTrajLikSmokeTest(unittest.TestCase):
    def _write_cache(self, root):
        index = []
        for sample_index in range(4):
            filename = f"{sample_index}.pt"
            states = torch.randn(4, 8, 2, 2)
            torch.save(
                {
                    "z_0": states[0],
                    "z_seq": states[1:],
                    "eps_seq": torch.randn(3, 8, 2, 2),
                    "delta_z_seq": states[1:] - states[:-1],
                    "a_end_coarse": torch.linalg.vector_norm(states[-1], dim=0),
                    "split": "train",
                    "is_normal": True,
                },
                root / filename,
            )
            index.append(
                {
                    "file": filename,
                    "category": "a",
                    "split": "train",
                    "is_normal": True,
                }
            )
        (root / "cache_meta.json").write_text(
            json.dumps(
                {
                    "num_images": 4,
                    "num_steps": 3,
                    "output_channels": 8,
                    "normal_only": True,
                    "config_sha256": "test",
                }
            ),
            encoding="utf-8",
        )
        (root / "cache_index.json").write_text(json.dumps(index), encoding="utf-8")

    def test_one_epoch_checkpoint_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            self._write_cache(cache_dir)
            checkpoint_path = root / "head.pth"
            args = Namespace(
                cache_dir=str(cache_dir),
                output_path=str(checkpoint_path),
                device="cpu",
                epochs=1,
                batch_size=2,
                num_workers=0,
                learning_rate=1e-4,
                weight_decay=1e-4,
                grad_clip=1.0,
                calibration_fraction=0.25,
                seed=42,
                projection_dim=4,
                token_dim=8,
                trajectory_dim=4,
                dcte_layers=1,
                dcte_heads=2,
                flow_blocks=1,
                flow_bins=4,
                lambda_msm=1.0,
            )

            train(args)
            head, calibrator, checkpoint = load_trajlik_checkpoint(
                checkpoint_path
            )

            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue(calibrator.fitted)
            self.assertFalse(head.training)
            self.assertEqual(checkpoint["calibration_indices"].numel(), 1)


if __name__ == "__main__":
    unittest.main()
