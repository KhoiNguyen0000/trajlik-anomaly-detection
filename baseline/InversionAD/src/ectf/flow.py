import math

import torch
from torch import Tensor, nn

from .conditioner import EndpointConditioner
from .splines import rational_quadratic_spline


class InvertibleLinear(nn.Module):
    """Trainable LU-parameterized linear mixing with exact log determinant."""

    def __init__(self, features: int):
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        self.features = features
        permutation = torch.eye(features)[torch.randperm(features)]
        self.register_buffer("permutation", permutation)
        self.lower = nn.Parameter(torch.zeros(features, features))
        self.upper = nn.Parameter(torch.zeros(features, features))
        self.log_diagonal = nn.Parameter(torch.zeros(features))

    def weight(self) -> Tensor:
        identity = torch.eye(
            self.features,
            device=self.lower.device,
            dtype=self.lower.dtype,
        )
        lower = identity + torch.tril(self.lower, diagonal=-1)
        upper = torch.triu(self.upper, diagonal=1) + torch.diag(
            torch.exp(self.log_diagonal)
        )
        return self.permutation @ lower @ upper

    def forward(self, inputs: Tensor, inverse: bool = False) -> tuple[Tensor, Tensor]:
        weight = self.weight()
        if inverse:
            outputs = torch.linalg.solve(weight, inputs.T).T
            logabsdet = -self.log_diagonal.sum()
        else:
            outputs = inputs @ weight.T
            logabsdet = self.log_diagonal.sum()
        return outputs, logabsdet.expand(inputs.shape[0])


class ConditionalSplineCoupling(nn.Module):
    def __init__(
        self,
        features: int,
        condition_dim: int,
        mask: Tensor,
        hidden_dim: int = 128,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        if mask.dtype != torch.bool or mask.shape != (features,):
            raise ValueError("mask must be a boolean vector of length features")
        if mask.all() or (~mask).all():
            raise ValueError("mask must preserve and transform at least one feature")

        self.features = features
        self.condition_dim = condition_dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_derivative = min_derivative
        self.register_buffer("mask", mask)

        identity_features = int(mask.sum())
        transformed_features = features - identity_features
        parameters_per_feature = 3 * num_bins - 1
        self.transformed_features = transformed_features
        self.parameter_network = nn.Sequential(
            nn.Linear(identity_features + condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                transformed_features * parameters_per_feature,
            ),
        )
        final_layer = self.parameter_network[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        derivative_bias = math.log(math.expm1(1.0 - min_derivative))
        bias = final_layer.bias.view(transformed_features, parameters_per_feature)
        with torch.no_grad():
            bias[:, 2 * num_bins :] = derivative_bias

    def forward(
        self,
        inputs: Tensor,
        condition: Tensor,
        inverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 2 or inputs.shape[1] != self.features:
            raise ValueError("inputs must have shape [N,features]")
        if condition.shape != (inputs.shape[0], self.condition_dim):
            raise ValueError("condition shape does not match inputs")

        identity = inputs[:, self.mask]
        transformed = inputs[:, ~self.mask]
        parameters = self.parameter_network(
            torch.cat((identity, condition), dim=-1)
        ).reshape(inputs.shape[0], self.transformed_features, -1)
        widths, heights, derivatives = torch.split(
            parameters,
            (self.num_bins, self.num_bins, self.num_bins - 1),
            dim=-1,
        )
        transformed, logabsdet = rational_quadratic_spline(
            transformed,
            widths,
            heights,
            derivatives,
            inverse=inverse,
            tail_bound=self.tail_bound,
            min_derivative=self.min_derivative,
        )
        outputs = inputs.clone()
        outputs[:, ~self.mask] = transformed
        return outputs, logabsdet.sum(dim=-1)


class ConditionalFlowStep(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mixing = InvertibleLinear(kwargs["features"])
        self.coupling = ConditionalSplineCoupling(*args, **kwargs)

    def forward(
        self,
        inputs: Tensor,
        condition: Tensor,
        inverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if inverse:
            outputs, coupling_logdet = self.coupling(
                inputs,
                condition,
                inverse=True,
            )
            outputs, mixing_logdet = self.mixing(outputs, inverse=True)
        else:
            outputs, mixing_logdet = self.mixing(inputs)
            outputs, coupling_logdet = self.coupling(outputs, condition)
        return outputs, mixing_logdet + coupling_logdet


class EndpointConditionedTrajectoryFlow(nn.Module):
    """Conditional spline flow implementing TrajLik-AD Module 2."""

    def __init__(
        self,
        trajectory_dim: int = 64,
        z0_dim: int = 272,
        condition_dim: int = 64,
        global_dim: int = 32,
        position_dim: int = 16,
        conditioner_hidden_dim: int = 128,
        coupling_hidden_dim: int = 128,
        num_blocks: int = 4,
        num_bins: int = 8,
        tail_bound: float = 3.0,
    ):
        super().__init__()
        if trajectory_dim < 2:
            raise ValueError("trajectory_dim must be at least two")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.trajectory_dim = trajectory_dim
        self.condition_dim = condition_dim
        self.conditioner = EndpointConditioner(
            z0_dim=z0_dim,
            global_dim=global_dim,
            position_dim=position_dim,
            condition_dim=condition_dim,
            hidden_dim=conditioner_hidden_dim,
        )
        base_mask = torch.arange(trajectory_dim) % 2 == 0
        self.blocks = nn.ModuleList(
            ConditionalFlowStep(
                features=trajectory_dim,
                condition_dim=condition_dim,
                mask=base_mask if index % 2 == 0 else ~base_mask,
                hidden_dim=coupling_hidden_dim,
                num_bins=num_bins,
                tail_bound=tail_bound,
            )
            for index in range(num_blocks)
        )

    def transform(
        self,
        inputs: Tensor,
        condition: Tensor,
        *,
        inverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.trajectory_dim:
            raise ValueError(
                "inputs must have shape [B,P,trajectory_dim]"
            )
        if condition.shape != (*inputs.shape[:2], self.condition_dim):
            raise ValueError("condition shape does not match trajectory inputs")

        batch, patches, _ = inputs.shape
        values = inputs.reshape(batch * patches, self.trajectory_dim)
        flat_condition = condition.reshape(batch * patches, self.condition_dim)
        total_logdet = values.new_zeros(values.shape[0])
        blocks = reversed(self.blocks) if inverse else self.blocks
        for block in blocks:
            values, logdet = block(
                values,
                flat_condition,
                inverse=inverse,
            )
            total_logdet = total_logdet + logdet
        return (
            values.reshape(batch, patches, self.trajectory_dim),
            total_logdet.reshape(batch, patches),
        )

    def forward(
        self,
        trajectory_codes: Tensor,
        endpoint_energy: Tensor,
        z0: Tensor,
    ) -> dict[str, Tensor]:
        condition = self.conditioner(endpoint_energy, z0)
        base_latents, log_det = self.transform(trajectory_codes, condition)
        base_log_prob = -0.5 * (
            base_latents.square() + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        log_prob = base_log_prob + log_det
        return {
            "path_nll": -log_prob,
            "log_prob": log_prob,
            "base_latents": base_latents,
            "log_det": log_det,
            "condition": condition,
        }

    def inverse(
        self,
        base_latents: Tensor,
        endpoint_energy: Tensor,
        z0: Tensor,
    ) -> tuple[Tensor, Tensor]:
        condition = self.conditioner(endpoint_energy, z0)
        return self.transform(base_latents, condition, inverse=True)

    def nll_loss(
        self,
        trajectory_codes: Tensor,
        endpoint_energy: Tensor,
        z0: Tensor,
    ) -> Tensor:
        return self(trajectory_codes, endpoint_energy, z0)["path_nll"].mean()
