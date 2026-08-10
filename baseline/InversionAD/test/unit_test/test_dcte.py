import unittest

import torch

from src.dcte import DCTE, MSMLoss


class DCTETest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        self.batch_size = 2
        self.module0_output = {
            "states": torch.randn(
                self.batch_size,
                4,
                272,
                16,
                16,
            ),
            "epsilons": torch.randn(
                self.batch_size,
                3,
                272,
                16,
                16,
            ),
            "a_end": torch.rand(
                self.batch_size,
                16,
                16,
            ),
        }

        self.dcte = DCTE(
            input_dim=272,
            projection_dim=64,
            token_dim=128,
            trajectory_dim=64,
            num_steps=3,
            num_heads=4,
            num_layers=2,
            feedforward_dim=512,
            dropout=0.0,
        )

    def test_forward_without_masking(self):
        output = self.dcte(
            self.module0_output,
            mask=False,
        )

        self.assertEqual(
            output["trajectory_codes"].shape,
            (self.batch_size, 256, 64),
        )
        self.assertEqual(
            output["step_tokens"].shape,
            (self.batch_size, 256, 3, 128),
        )
        self.assertEqual(
            output["state_prev_p"].shape,
            (self.batch_size, 256, 3, 272),
        )
        self.assertEqual(
            output["epsilon_p"].shape,
            (self.batch_size, 256, 3, 272),
        )
        self.assertEqual(
            output["delta_p"].shape,
            (self.batch_size, 256, 3, 272),
        )
        self.assertIsNone(
            output["masked_step_indices"]
        )
        self.assertNotIn(
            "masked_step_tokens",
            output,
        )

    def test_forward_accepts_raw_module0_output(self):
        states = self.module0_output["states"]
        epsilons = self.module0_output["epsilons"]
        raw_output = {
            "z_0": states[:, 0],
            "z_seq": list(states[:, 1:].unbind(dim=1)),
            "eps_seq": list(epsilons.unbind(dim=1)),
            "delta_z_seq": list(
                (states[:, 1:] - states[:, :-1]).unbind(dim=1)
            ),
        }

        output = self.dcte(raw_output, mask=False)

        self.assertEqual(
            output["trajectory_codes"].shape,
            (self.batch_size, 256, 64),
        )

    def test_forward_with_masking(self):
        output = self.dcte(
            self.module0_output,
            mask=True,
        )

        self.assertEqual(
            output["masked_step_indices"].shape,
            (self.batch_size, 256),
        )
        self.assertEqual(
            output["masked_step_tokens"].shape,
            (self.batch_size, 256, 3, 128),
        )
        self.assertTrue(
            (output["masked_step_indices"] >= 0).all()
        )
        self.assertTrue(
            (output["masked_step_indices"] < 3).all()
        )

    def test_msm_loss_backward(self):
        msm_loss = MSMLoss(
            input_dim=272,
            projection_dim=64,
            token_dim=128,
            trajectory_dim=64,
            lambda_cos=1.0,
        )

        output = self.dcte(
            self.module0_output,
            mask=True,
        )

        loss = msm_loss(
            output,
            self.dcte.tokenizer.step_embedding,
        )

        self.assertTrue(torch.isfinite(loss))

        loss.backward()

        gradient = (
            self.dcte
            .cross_step_encoder
            .trajectory_projection
            .weight
            .grad
        )

        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_zero_displacement_is_finite(self):
        module0_output = {
            "states": torch.zeros(
                self.batch_size,
                4,
                272,
                16,
                16,
            ),
            "epsilons": torch.zeros(
                self.batch_size,
                3,
                272,
                16,
                16,
            ),
            "a_end": torch.zeros(
                self.batch_size,
                16,
                16,
            ),
        }

        output = self.dcte(
            module0_output,
            mask=False,
        )

        self.assertTrue(
            torch.isfinite(output["step_tokens"]).all()
        )
        self.assertTrue(
            torch.isfinite(output["trajectory_codes"]).all()
        )


if __name__ == "__main__":
    unittest.main()
