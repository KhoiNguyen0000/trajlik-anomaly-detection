import torch.nn as nn

from .encoder import CrossStepEncoder
from .masking import MaskedStepModule
from .tokenizer import TrajectoryTokenizer


class DCTE(nn.Module):
    def __init__(
        self,
        input_dim=272,
        projection_dim=64,
        token_dim=128,
        trajectory_dim=64,
        num_steps=3,
        num_heads=4,
        num_layers=2,
        feedforward_dim=512,
        dropout=0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.token_dim = token_dim
        self.trajectory_dim = trajectory_dim
        self.num_steps = num_steps

        # =========================================
        # Trajectory tokenizer
        # =========================================

        self.tokenizer = TrajectoryTokenizer(
            input_dim=input_dim,
            projection_dim=projection_dim,
            token_dim=token_dim,
            num_steps=num_steps,
        )

        # =========================================
        # Mask module
        # =========================================

        self.masked_step_module = MaskedStepModule(
            token_dim=token_dim,
            num_steps=num_steps,
        )

        # =========================================
        # Cross-step encoder
        # =========================================

        self.cross_step_encoder = CrossStepEncoder(
            token_dim=token_dim,
            trajectory_dim=trajectory_dim,
            num_steps=num_steps,
            num_heads=num_heads,
            num_layers=num_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

    def forward(
        self,
        module0_output,
        mask=False,
    ):
        """
        Args:
            module0_output:
                output from Module 0

            mask:
                True:
                    randomly mask one step per patch

                False:
                    use complete trajectory

        Returns:
            Dictionary containing trajectory representation
            and tensors required by later training objectives.
        """

        # =========================================
        # 1. Tokenize trajectory
        # =========================================

        tokenizer_output = self.tokenizer(
            module0_output
        )

        step_tokens = tokenizer_output[
            "step_tokens"
        ]
        # [B, P, S, token_dim]

        state_prev_p = tokenizer_output[
            "state_prev_p"
        ]
        # [B, P, S, input_dim]

        epsilon_p = tokenizer_output[
            "epsilon_p"
        ]
        # [B, P, S, input_dim]

        delta_p = tokenizer_output[
            "delta_p"
        ]
        # [B, P, S, input_dim]

        # =========================================
        # 2. Masked trajectory
        # =========================================

        if mask:

            (
                encoder_input,
                masked_step_indices,
            ) = self.masked_step_module(
                step_tokens,
                self.tokenizer.step_embedding,
            )

            # encoder_input:
            # [B, P, S, token_dim]
            #
            # masked_step_indices:
            # [B, P]

        # =========================================
        # 3. Full trajectory
        # =========================================

        else:

            encoder_input = step_tokens

            masked_step_indices = None

        # =========================================
        # 4. Cross-step encoding
        # =========================================

        trajectory_codes = (
            self.cross_step_encoder(
                encoder_input
            )
        )
        # [B, P, trajectory_dim]

        # =========================================
        # 5. Return representation
        # =========================================

        output = {
            "trajectory_codes":
                trajectory_codes,

            "step_tokens":
                step_tokens,

            "state_prev_p":
                state_prev_p,

            "epsilon_p":
                epsilon_p,

            "delta_p":
                delta_p,
            "masked_step_indices":
                masked_step_indices,
        }

        # Only exists when masking is enabled
        if mask:
            output["masked_step_tokens"] = (
                encoder_input
            )

        return output
