import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedStepModule(nn.Module):
    def __init__(
        self,
        token_dim=128,
        num_steps=3,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.num_steps = num_steps

        # Learnable mask token
        self.mask_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                1,
                token_dim,
            )
        )

        nn.init.normal_(
            self.mask_token,
            mean=0.0,
            std=0.02,
        )

    def forward(
        self,
        step_tokens,
        step_embedding,
    ):
        """
        step_tokens:
            [B, P, S, D]

        step_embedding:
            [1, 1, S, D]

        Returns:
            masked_step_tokens:
                [B, P, S, D]

            masked_step_indices:
                [B, P]
        """

        B, P, S, D = step_tokens.shape

        # -----------------------------------------
        # Randomly choose one step for each patch
        # -----------------------------------------

        masked_step_indices = torch.randint(
            low=0,
            high=S,
            size=(B, P),
            device=step_tokens.device,
        )
        # [B, P]

        # -----------------------------------------
        # Convert indices -> binary mask
        # -----------------------------------------

        step_mask = F.one_hot(
            masked_step_indices,
            num_classes=S,
        ).bool()
        # [B, P, S]

        step_mask = step_mask.unsqueeze(-1)
        # [B, P, S, 1]

        # -----------------------------------------
        # Replace content but preserve step identity
        # -----------------------------------------

        masked_value = (
            self.mask_token
            + step_embedding
        )
        # [1, 1, S, D]

        masked_step_tokens = torch.where(
            step_mask,
            masked_value,
            step_tokens,
        )
        # [B, P, S, D]

        return (
            masked_step_tokens,
            masked_step_indices,
        )
