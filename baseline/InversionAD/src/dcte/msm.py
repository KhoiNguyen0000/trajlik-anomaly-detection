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

        # Qz
        self.state_norm = nn.LayerNorm(input_dim)
        self.state_projection = nn.Linear(
            input_dim,
            projection_dim,
        )

        # Qepsilon
        self.epsilon_norm = nn.LayerNorm(input_dim)
        self.epsilon_projection = nn.Linear(
            input_dim,
            projection_dim,
        )

        # Qdelta
        self.delta_norm = nn.LayerNorm(input_dim)
        self.delta_projection = nn.Linear(
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

        state_prev_p = dcte_output["state_prev_p"]
        # [B,P,S,272]

        epsilon_p = dcte_output["epsilon_p"]
        # [B,P,S,272]

        delta_p = dcte_output["delta_p"]
        # [B,P,S,272]

        B, P = m.shape

        # ====================================
        # Get target information at step m
        # ====================================

        state_m = self.gather_step(
            state_prev_p,
            m,
        )
        # [B,P,272]

        epsilon_m = self.gather_step(
            epsilon_p,
            m,
        )
        # [B,P,272]

        delta_m = self.gather_step(
            delta_p,
            m,
        )
        # [B,P,272]

        # ====================================
        # y_{p,m}
        # ====================================

        target_state = self.state_projection(
            self.state_norm(state_m)
        )
        # [B,P,64]

        target_epsilon = self.epsilon_projection(
            self.epsilon_norm(epsilon_m)
        )
        # [B,P,64]

        target_delta = self.delta_projection(
            self.delta_norm(delta_m)
        )
        # [B,P,64]

        target = torch.cat(
            [
                target_state,
                target_epsilon,
                target_delta,
            ],
            dim=-1,
        )
        # [B,P,192]

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
