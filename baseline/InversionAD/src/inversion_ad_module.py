import torch
import torch.nn.functional as F
from torch import Tensor, nn


class InversionADModule(nn.Module):
    """Run feature extraction, DDIM inversion, and endpoint scoring."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        eval_denoiser: nn.Module,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.eval_denoiser = eval_denoiser

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        labels: Tensor | None = None,
    ) -> dict:
        height, width = images.shape[-2:]

        z_0, _ = self.feature_extractor(images)

        start_t = torch.zeros(
            z_0.shape[0],
            dtype=torch.long,
            device=z_0.device,
        )

        with torch.amp.autocast(
            device_type=images.device.type,
            dtype=torch.bfloat16,
            enabled=images.is_cuda,
        ):
            final_latent, z_seq, eps_seq, delta_z_seq = (
                self.eval_denoiser.ddim_reverse_sample(
                    z_0,
                    start_t,
                    labels,
                    eta=0.0,
                    return_intermediates=True,
                )
            )

        latents_l2 = torch.sum(final_latent**2, dim=1).sqrt()
        a_end = F.interpolate(
            latents_l2.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return {
            "z_0": z_0,
            "z_seq": z_seq,
            "eps_seq": eps_seq,
            "delta_z_seq": delta_z_seq,
            "a_end": a_end,
        }
