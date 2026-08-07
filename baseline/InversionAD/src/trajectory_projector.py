import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List


class TrajectoryProjector(nn.Module):
    """Low-rank linear projections + FP16 casting for trajectory caching.
    
    Projects z, ε, Δz tokens from full channel dim -> reduced dim,
    then casts to FP16 for storage savings.
    """

    def __init__(self,
                 in_channels: int = 272,
                 proj_dim: int = 68,
                 projection: str = 'linear',
                 storage_dtype: torch.dtype = torch.float16):
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
            self.proj_z = nn.Linear(
                in_channels, proj_dim, bias=False
            )
            self.proj_eps = nn.Linear(
                in_channels, proj_dim, bias=False
            )
            self.proj_delta = nn.Linear(
                in_channels, proj_dim, bias=False
            )
        else:
            self.proj_z = nn.Identity()
            self.proj_eps = nn.Identity()
            self.proj_delta = nn.Identity()

    @torch.no_grad()
    def project_and_compress(
            self,
            z_0: Tensor,  # (B, C, H, W)
            z_seq: List[Tensor],  # S x (B, C, H, W)
            eps_seq: List[Tensor],
            delta_z_seq: List[Tensor],
    ) -> Dict[str, Tensor]:
        """Project + cast FP16.
        
        Returns dict with keys: z_0, z_seq, eps_seq, delta_z_seq
        All tensors are in FP16, with channel dim = proj_dim.
        """
        B, C, H, W = z_0.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        def _project(
                tensor: Tensor,
                proj_layer: nn.Module,
        ) -> Tensor:
            # (B,C,H,W) -> (B,H,W,C)
            tensor = tensor.permute(0, 2, 3, 1)

            # Linear or Identity
            tensor = proj_layer(tensor)

            # (B,H,W,C_out) -> (B,C_out,H,W)
            tensor = tensor.permute(0, 3, 1, 2)

            return tensor.to(dtype=self.storage_dtype)

        # Project z_0
        z_0_proj = _project(z_0, self.proj_z).to(torch.float16)

        # Project sequences
        z_seq_proj = []
        eps_seq_proj = []
        delta_z_seq_proj = []

        for i in range(len(z_seq)):
            z_seq_proj.append(_project(z_seq[i], self.proj_z).to(dtype=self.storage_dtype))
            eps_seq_proj.append(_project(eps_seq[i], self.proj_eps).to(dtype=self.storage_dtype))
            delta_z_seq_proj.append(_project(delta_z_seq[i], self.proj_delta).to(dtype=self.storage_dtype))

        # Stack sequences -> (B, S, proj_dim, H, W)
        z_seq_stacked = torch.stack(z_seq_proj, dim=1) if len(z_seq_proj) > 0 else torch.empty(
            (B, 0, self.proj_dim, H, W), dtype=self.storage_dtype, device=z_0.device)
        eps_seq_stacked = torch.stack(eps_seq_proj, dim=1) if len(eps_seq_proj) > 0 else torch.empty(
            (B, 0, self.proj_dim, H, W), dtype=self.storage_dtype, device=z_0.device)
        delta_z_seq_stacked = torch.stack(delta_z_seq_proj, dim=1) if len(delta_z_seq_proj) > 0 else torch.empty(
            (B, 0, self.proj_dim, H, W), dtype=self.storage_dtype, device=z_0.device)

        return {
            "z_0": z_0_proj,
            "z_seq": z_seq_stacked,
            "eps_seq": eps_seq_stacked,
            "delta_z_seq": delta_z_seq_stacked,
        }
