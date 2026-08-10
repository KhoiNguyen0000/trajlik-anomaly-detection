import torch
import torch.nn as nn

from src.trajectory_batch import build_trajectory_batch


class TrajectoryTokenizer(nn.Module):
    def __init__(
        self,
        input_dim=272,
        projection_dim=64,
        token_dim=128,
        num_steps=3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.token_dim = token_dim
        self.num_steps = num_steps

        # Normalize state and epsilon features
        self.state_norm = nn.LayerNorm(input_dim)
        self.epsilon_norm = nn.LayerNorm(input_dim)

        # State projection
        self.state_projection = nn.Linear(
            input_dim,
            projection_dim,
        )

        # Epsilon projection
        self.epsilon_projection = nn.Linear(
            input_dim,
            projection_dim,
        )

        # Direction projection
        self.direction_projection = nn.Linear(
            input_dim,
            projection_dim,
        )

        self.compressed_feature_dim = 3 * projection_dim + 1

        # Project compressed features into token space
        self.step_token_projection = nn.Linear(
            self.compressed_feature_dim,
            token_dim,
        )

        # Learnable step embedding
        self.step_embedding = nn.Parameter(
            torch.zeros(
                1,
                1,
                num_steps,
                token_dim,
            )
        )

        nn.init.normal_(
            self.step_embedding,
            mean=0.0,
            std=0.02,
        )

    @staticmethod
    def to_patch_sequence(x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [B, S, C, H, W]

        Output:
            x: [B, P, S, C]
            where P = H * W
        """
        B, S, C, H, W = x.shape

        return (
            x.permute(0, 3, 4, 1, 2)
            .reshape(B, H * W, S, C)
        )

    def forward(self, module0_output):
        trajectory_batch = build_trajectory_batch(module0_output)
        states = trajectory_batch["states"]
        epsilons = trajectory_batch["epsilons"]

        if states.shape[1] != self.num_steps + 1:
            raise ValueError(
                f"Expected {self.num_steps} trajectory steps, "
                f"got {states.shape[1] - 1}"
            )

        if states.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} channels, got {states.shape[2]}"
            )

        # Trajectory tokenization
        state_prev = states[:, :-1]       # [B, 3, 272, 16, 16]
        state_next = states[:, 1:]        # [B, 3, 272, 16, 16]
        deltas = state_next - state_prev  # [B, 3, 272, 16, 16]

        magnitude = torch.linalg.vector_norm(
            deltas,
            ord=2,
            dim=2,
            keepdim=True,
        )  # [B, 3, 1, 16, 16]

        direction = deltas / (magnitude + 1e-6)
        # [B, 3, 272, 16, 16]

        log_magnitude = torch.log(magnitude + 1e-6)
        # [B, 3, 1, 16, 16]

        delta_p = self.to_patch_sequence(deltas)
        # [B, 256, 3, 272]

        state_prev_p = self.to_patch_sequence(state_prev)
        # [B, 256, 3, 272]

        epsilon_p = self.to_patch_sequence(epsilons)
        # [B, 256, 3, 272]

        direction_p = self.to_patch_sequence(direction)
        # [B, 256, 3, 272]

        log_magnitude_p = self.to_patch_sequence(log_magnitude)
        # [B, 256, 3, 1]

        # Projection
        state_feature = self.state_projection(
            self.state_norm(state_prev_p)
        )

        epsilon_feature = self.epsilon_projection(
            self.epsilon_norm(epsilon_p)
        )

        direction_feature = self.direction_projection(
            direction_p
        )

        # Concatenate compressed trajectory features
        compressed_step_feature = torch.cat(
            [
                state_feature,
                epsilon_feature,
                direction_feature,
                log_magnitude_p,
            ],
            dim=-1,
        )

        # Generate step tokens
        step_tokens = self.step_token_projection(
            compressed_step_feature
        )

        step_tokens = step_tokens + self.step_embedding

        return {
            "step_tokens": step_tokens,
            "state_prev_p": state_prev_p,
            "epsilon_p": epsilon_p,
            "delta_p": delta_p,
        }
