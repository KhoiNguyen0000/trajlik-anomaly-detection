import torch
import torch.nn as nn


class CrossStepEncoder(nn.Module):
    def __init__(
        self,
        token_dim=128,
        trajectory_dim=64,
        num_steps=3,
        num_heads=4,  # 4 heads
        num_layers=2,  # 2 blocks
        feedforward_dim=512,
        dropout=0.0,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.trajectory_dim = trajectory_dim
        self.num_steps = num_steps

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN
        )

        # Cross-step Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(token_dim),
            enable_nested_tensor=False,
        )

        # Project pooled step features into trajectory space
        self.trajectory_projection = nn.Linear(
            token_dim,
            trajectory_dim,
        )

        # Normalize the final trajectory code
        self.trajectory_norm = nn.LayerNorm(
            trajectory_dim
        )

    def forward(
        self,
        step_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
            step_tokens: [B, P, S, token_dim]

        Output:
            trajectory_codes: [B, P, trajectory_dim]
        """
        B, P, S, D = step_tokens.shape

        # Treat each patch trajectory as an independent sequence
        step_tokens = step_tokens.reshape(
            B * P,
            S,
            D,
        )
        # [B * P, S, token_dim]

        # Encode interactions across inversion steps
        encoded_steps = self.transformer_encoder(
            step_tokens
        )
        # [B * P, S, token_dim]

        # Aggregate all step representations
        pooled_trajectory = encoded_steps.mean(
            dim=1
        )
        # [B * P, token_dim]

        # Generate the final trajectory code
        trajectory_codes = self.trajectory_projection(
            pooled_trajectory
        )
        # [B * P, trajectory_dim]

        trajectory_codes = self.trajectory_norm(
            trajectory_codes
        )
        # [B * P, trajectory_dim]

        # Restore batch and patch dimensions
        trajectory_codes = trajectory_codes.reshape(
            B,
            P,
            self.trajectory_dim,
        )
        # [B, P, trajectory_dim]

        return trajectory_codes
