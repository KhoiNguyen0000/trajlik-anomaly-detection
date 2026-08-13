import torch
import torch.nn as nn
import torch.nn.functional as F


class MSMLoss(nn.Module):
    def __init__(
        self,
        input_dim=272,
        projection_dim=64,
        token_dim=128,
        trajectory_dim=64,
        lambda_cos=1.0,
    ):
        super().__init__()

        self.lambda_cos = lambda_cos

        # The MSM target encoder is a fixed teacher. If Qz/Qepsilon/Qdelta
        # are trainable, they can collapse their targets toward zero together
        # with the decoder, minimizing MSM without preserving information.
        self.state_norm = nn.LayerNorm(
            input_dim,
            elementwise_affine=False,
        )
        self.state_projection = self._frozen_projection(
            input_dim,
            projection_dim,
        )

        self.epsilon_norm = nn.LayerNorm(
            input_dim,
            elementwise_affine=False,
        )
        self.epsilon_projection = self._frozen_projection(
            input_dim,
            projection_dim,
        )

        self.delta_norm = nn.LayerNorm(
            input_dim,
            elementwise_affine=False,
        )
        self.delta_projection = self._frozen_projection(
            input_dim,
            projection_dim,
        )

        target_dim = 3 * projection_dim

        # D_eta
        self.decoder = nn.Sequential(
            nn.Linear(
                trajectory_dim + token_dim,
                token_dim,
            ),
            nn.GELU(),
            nn.Linear(
                token_dim,
                target_dim,
            ),
        )

    @staticmethod
    def _frozen_projection(
        input_dim,
        projection_dim,
    ):
        projection = nn.Linear(
            input_dim,
            projection_dim,
            bias=False,
        )
        nn.init.orthogonal_(projection.weight)
        projection.requires_grad_(False)
        return projection

    @staticmethod
    def gather_step(
        x,
        step_indices,
    ):
        """
        x:
            [B, P, S, C]

        step_indices:
            [B, P]

        return:
            [B, P, C]
        """

        C = x.shape[-1]

        index = (
            step_indices
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        # [B,P,1,1]

        index = index.expand(
            -1,
            -1,
            1,
            C,
        )
        # [B,P,1,C]

        return torch.gather(
            x,
            dim=2,
            index=index,
        ).squeeze(2)
        # [B,P,C]

    @torch.no_grad()
    def build_target(self, dcte_output):
        """Build the fixed compressed target y_{p,m}."""

        masked_step_indices = dcte_output["masked_step_indices"]
        if masked_step_indices is None:
            raise ValueError("MSM requires DCTE output with masking enabled")

        state_m = self.gather_step(
            dcte_output["state_prev_p"],
            masked_step_indices,
        )
        epsilon_m = self.gather_step(
            dcte_output["epsilon_p"],
            masked_step_indices,
        )
        delta_m = self.gather_step(
            dcte_output["delta_p"],
            masked_step_indices,
        )

        target_state = self.state_projection(
            self.state_norm(state_m)
        )
        target_epsilon = self.epsilon_projection(
            self.epsilon_norm(epsilon_m)
        )
        target_delta = self.delta_projection(
            self.delta_norm(delta_m)
        )

        return torch.cat(
            [
                target_state,
                target_epsilon,
                target_delta,
            ],
            dim=-1,
        )

    def forward(
        self,
        dcte_output,
        step_embedding,
    ):

        # ====================================
        # Get DCTE outputs
        # ====================================

        h = dcte_output["trajectory_codes"]
        # [B,P,64]

        m = dcte_output["masked_step_indices"]
        # [B,P]

        target = self.build_target(dcte_output)
        # [B,P,192]

        B, P = m.shape

        # ====================================
        # Get e(tau_m)
        # ====================================

        step_embedding = step_embedding.expand(
            B,
            P,
            -1,
            -1,
        )
        # [B,P,S,128]

        e_m = self.gather_step(
            step_embedding,
            m,
        )
        # [B,P,128]

        # ====================================
        # y_hat = D([h_p ; e_m])
        # ====================================

        decoder_input = torch.cat(
            [
                h,
                e_m,
            ],
            dim=-1,
        )
        # [B,P,192]

        prediction = self.decoder(
            decoder_input
        )
        # [B,P,192]

        # ====================================
        # Loss
        # ====================================

        huber_loss = F.smooth_l1_loss(
            prediction,
            target,
        )

        cosine_loss = (
            1.0
            - F.cosine_similarity(
                prediction,
                target,
                dim=-1,
            )
        ).mean()

        loss = (
            huber_loss
            + self.lambda_cos * cosine_loss
        )

        return loss
