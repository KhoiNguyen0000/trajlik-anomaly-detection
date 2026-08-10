import unittest

import torch
from torch import nn

from src.inversion_ad_module import InversionADModule


class FeatureExtractorStub(nn.Module):
    def forward(self, images):
        features = images[:, :2, ::2, ::2]
        return features, None


class DenoiserStub(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def ddim_reverse_sample(
        self,
        z_0,
        start_t,
        labels,
        eta,
        return_intermediates,
    ):
        self.start_t = start_t
        states = [z_0 + step for step in (1.0, 2.0, 3.0)]
        epsilons = [torch.zeros_like(z_0) for _ in range(3)]
        deltas = [torch.ones_like(z_0) for _ in range(3)]
        return states[-1], states, epsilons, deltas


class InversionADModuleTest(unittest.TestCase):
    def test_cpu_output_exposes_coarse_and_pixel_endpoint_maps(self):
        denoiser = DenoiserStub()
        module = InversionADModule(FeatureExtractorStub(), denoiser)
        images = torch.randn(2, 3, 8, 8)

        output = module(images)

        self.assertEqual(tuple(denoiser.start_t.shape), (2,))
        self.assertEqual(tuple(output["a_end_coarse"].shape), (2, 4, 4))
        self.assertEqual(tuple(output["a_end"].shape), (2, 8, 8))
        self.assertEqual(output["a_end_coarse"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(output["a_end"]).all())


if __name__ == "__main__":
    unittest.main()
