from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor


class TrajectoryProjector(nn.Module):
    """Optionally project trajectory channels and cast them for storage."""

    def __init__(
        self,
        in_channels: int = 272,
        proj_dim: int = 68,
        projection: str = "linear",
        storage_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        if projection not in {"none", "linear"}:
            raise ValueError(
                f"Unsupported projection mode: {projection}"
            )

        if storage_dtype not in {torch.float16, torch.float32}:
            raise ValueError(
                f"Unsupported storage dtype: {storage_dtype}"
            )

        self.in_channels = in_channels
        self.proj_dim = (
            proj_dim if projection == "linear" else in_channels
        )
        self.projection = projection
        self.storage_dtype = storage_dtype

        if projection == "linear":
            self.proj_z = nn.Linear(in_channels, proj_dim, bias=False)
            self.proj_eps = nn.Linear(in_channels, proj_dim, bias=False)
            self.proj_delta = nn.Linear(in_channels, proj_dim, bias=False)
        else:
            self.proj_z = nn.Identity()
            self.proj_eps = nn.Identity()
            self.proj_delta = nn.Identity()

    @torch.no_grad()
    def project_and_compress(
        self,
        z_0: Tensor,
        z_seq: List[Tensor],
        eps_seq: List[Tensor],
        delta_z_seq: List[Tensor],
    ) -> Dict[str, Tensor]:
        """Return storage-ready tensors shaped `(B, S, C, H, W)`."""
        B, C, H, W = z_0.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {C}"
            )
        sequence_lengths = {len(z_seq), len(eps_seq), len(delta_z_seq)}
        if len(sequence_lengths) != 1:
            raise ValueError("Trajectory sequences must have the same length")

        def project(tensor: Tensor, layer: nn.Module) -> Tensor:
            tensor = tensor.permute(0, 2, 3, 1)
            tensor = layer(tensor)
            tensor = tensor.permute(0, 3, 1, 2)
            return tensor.to(dtype=self.storage_dtype)

        def project_sequence(tensors: List[Tensor], layer: nn.Module) -> Tensor:
            if tensors:
                return torch.stack(
                    [project(tensor, layer) for tensor in tensors],
                    dim=1,
                )
            return torch.empty(
                (B, 0, self.proj_dim, H, W),
                dtype=self.storage_dtype,
                device=z_0.device,
            )

        return {
            "z_0": project(z_0, self.proj_z),
            "z_seq": project_sequence(z_seq, self.proj_z),
            "eps_seq": project_sequence(eps_seq, self.proj_eps),
            "delta_z_seq": project_sequence(delta_z_seq, self.proj_delta),
        }
