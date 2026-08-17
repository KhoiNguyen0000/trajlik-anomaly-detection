import math

import torch
from torch import Tensor, nn


def sinusoidal_2d_positions(
    height: int,
    width: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if height <= 0 or width <= 0:
        raise ValueError("Spatial dimensions must be positive")
    if dimension <= 0 or dimension % 4:
        raise ValueError("Position dimension must be a positive multiple of 4")

    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(
            dimension // 4,
            device=device,
            dtype=dtype,
        )
        / max(dimension // 4 - 1, 1)
    )
    return torch.cat(
        (
            torch.sin(grid_x[..., None] * frequencies),
            torch.cos(grid_x[..., None] * frequencies),
            torch.sin(grid_y[..., None] * frequencies),
            torch.cos(grid_y[..., None] * frequencies),
        ),
        dim=-1,
    ).reshape(height * width, dimension)


class EndpointConditioner(nn.Module):
    """Build c_p from scalar endpoint energy, GAP(z0), and 2-D position."""

    def __init__(
        self,
        z0_dim: int,
        global_dim: int = 32,
        position_dim: int = 16,
        condition_dim: int = 64,
        hidden_dim: int = 128,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        if z0_dim <= 0:
            raise ValueError("z0_dim must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.z0_dim = z0_dim
        self.global_dim = global_dim
        self.position_dim = position_dim
        self.condition_dim = condition_dim
        self.epsilon = epsilon

        self.global_projection = nn.Linear(z0_dim, global_dim)
        self.global_norm = nn.LayerNorm(global_dim)
        self.condition_network = nn.Sequential(
            nn.Linear(1 + global_dim + position_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, condition_dim),
        )

    @staticmethod
    def _flatten_endpoint(endpoint_energy: Tensor) -> tuple[Tensor, int, int]:
        if endpoint_energy.ndim == 3:
            batch, height, width = endpoint_energy.shape
            return endpoint_energy.reshape(batch, height * width), height, width
        if endpoint_energy.ndim == 2:
            batch, patches = endpoint_energy.shape
            side = math.isqrt(patches)
            if side * side != patches:
                raise ValueError(
                    "Flat endpoint energy requires a square patch grid"
                )
            return endpoint_energy, side, side
        raise ValueError("endpoint_energy must have shape [B,H,W] or [B,P]")

    def forward(self, endpoint_energy: Tensor, z0: Tensor) -> Tensor:
        if z0.ndim != 4 or z0.shape[1] != self.z0_dim:
            raise ValueError(
                f"z0 must have shape [B,{self.z0_dim},H,W], "
                f"got {tuple(z0.shape)}"
            )

        endpoint, height, width = self._flatten_endpoint(endpoint_energy)
        if endpoint.shape[0] != z0.shape[0]:
            raise ValueError("endpoint_energy and z0 batch sizes do not match")
        if endpoint.shape[1] != z0.shape[-2] * z0.shape[-1]:
            raise ValueError("endpoint_energy and z0 patch grids do not match")

        # Equation (16): neither endpoint nor initial features receive density
        # gradients. Only the conditioner parameters are optimized.
        endpoint_feature = torch.log(
            endpoint.detach().float().clamp_min(0.0) + self.epsilon
        ).unsqueeze(-1)
        global_context = z0.detach().float().mean(dim=(-2, -1))
        global_context = self.global_norm(
            self.global_projection(global_context)
        )
        global_context = global_context[:, None, :].expand(
            -1,
            endpoint.shape[1],
            -1,
        )
        positions = sinusoidal_2d_positions(
            height,
            width,
            self.position_dim,
            device=endpoint.device,
            dtype=endpoint_feature.dtype,
        )
        positions = positions.unsqueeze(0).expand(endpoint.shape[0], -1, -1)
        return self.condition_network(
            torch.cat((endpoint_feature, global_context, positions), dim=-1)
        )
