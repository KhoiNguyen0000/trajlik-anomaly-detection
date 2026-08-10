import unittest

import torch

from src.ectf import EndpointConditionedTrajectoryFlow
from src.ectf.splines import rational_quadratic_spline


class RationalQuadraticSplineTest(unittest.TestCase):
    def test_forward_inverse_and_logdet_are_consistent(self):
        torch.manual_seed(1)
        inputs = torch.linspace(-4.0, 4.0, 33).reshape(11, 3)
        widths = torch.randn(11, 3, 8)
        heights = torch.randn(11, 3, 8)
        derivatives = torch.randn(11, 3, 7)

        outputs, forward_logdet = rational_quadratic_spline(
            inputs,
            widths,
            heights,
            derivatives,
        )
        reconstructed, inverse_logdet = rational_quadratic_spline(
            outputs,
            widths,
            heights,
            derivatives,
            inverse=True,
        )

        torch.testing.assert_close(reconstructed, inputs, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            forward_logdet + inverse_logdet,
            torch.zeros_like(forward_logdet),
            atol=1e-4,
            rtol=1e-4,
        )


class ECTFTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch = 2
        self.height = 4
        self.width = 4
        self.codes = torch.randn(
            self.batch,
            self.height * self.width,
            8,
            requires_grad=True,
        )
        self.endpoint = torch.rand(
            self.batch,
            self.height,
            self.width,
            requires_grad=True,
        )
        self.z0 = torch.randn(
            self.batch,
            12,
            self.height,
            self.width,
            requires_grad=True,
        )
        self.flow = EndpointConditionedTrajectoryFlow(
            trajectory_dim=8,
            z0_dim=12,
            condition_dim=16,
            global_dim=8,
            position_dim=8,
            conditioner_hidden_dim=32,
            coupling_hidden_dim=32,
            num_blocks=2,
            num_bins=4,
        )

    def test_shapes_are_finite_and_nll_backpropagates_to_codes(self):
        output = self.flow(self.codes, self.endpoint, self.z0)

        self.assertEqual(tuple(output["path_nll"].shape), (2, 16))
        self.assertEqual(tuple(output["base_latents"].shape), (2, 16, 8))
        self.assertTrue(torch.isfinite(output["path_nll"]).all())

        output["path_nll"].mean().backward()
        self.assertIsNotNone(self.codes.grad)
        self.assertTrue(torch.isfinite(self.codes.grad).all())
        self.assertIsNone(self.endpoint.grad)
        self.assertIsNone(self.z0.grad)

    def test_flow_transform_is_invertible(self):
        condition = self.flow.conditioner(self.endpoint, self.z0)
        base, forward_logdet = self.flow.transform(self.codes, condition)
        reconstructed, inverse_logdet = self.flow.transform(
            base,
            condition,
            inverse=True,
        )

        torch.testing.assert_close(
            reconstructed,
            self.codes,
            atol=3e-5,
            rtol=3e-5,
        )
        torch.testing.assert_close(
            forward_logdet + inverse_logdet,
            torch.zeros_like(forward_logdet),
            atol=3e-5,
            rtol=3e-5,
        )

    def test_condition_rejects_full_endpoint_vector(self):
        full_endpoint = torch.randn(2, 8, 4, 4)

        with self.assertRaisesRegex(ValueError, "endpoint_energy"):
            self.flow(self.codes, full_endpoint, self.z0)


if __name__ == "__main__":
    unittest.main()
