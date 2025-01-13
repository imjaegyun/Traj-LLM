# models/multimodal_laplace_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalLaplaceDecoder(nn.Module):
    def __init__(self, input_dim=6, output_dim=2, num_modes=3):
        super().__init__()
        self.num_modes = num_modes
        self.output_dim = output_dim

        # (6 -> 3072) to match high_level_features
        self.reduce_lane_prob = nn.Linear(input_dim, 3072)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=3072,
            num_heads=4,
            batch_first=True
        )

        self.pi_layer = nn.Linear(3072, num_modes)
        self.mu_layer = nn.Linear(3072, num_modes * output_dim)
        self.b_layer = nn.Linear(3072, num_modes * output_dim)
        self.uncertainty_layer = nn.Linear(3072, num_modes * output_dim)

        # Ensure b is positive
        self.b_activation = nn.Softplus()

    def forward(self, high_level_features, lane_probabilities):
        """
        high_level_features: [B, seq_len, 3072]
        lane_probabilities:  [B, seq_len, 6]
        """
        # 1) Reduce lane probabilities to match high_level_features
        lane_prob_emb = self.reduce_lane_prob(lane_probabilities)  # [B, seq_len, 3072]

        # 2) Cross-Attention
        attn_output, _ = self.cross_attention(
            query=high_level_features,    # [B, seq_len, 3072]
            key=lane_prob_emb,           # [B, seq_len, 3072]
            value=lane_prob_emb
        )  # [B, seq_len, 3072]

        # 3) Compute pi, mu, b, uncertainty
        pi = F.softmax(self.pi_layer(attn_output), dim=-1)     # [B, seq_len, num_modes]
        mu = self.mu_layer(attn_output)                        # [B, seq_len, num_modes * 2]
        b  = self.b_activation(self.b_layer(attn_output))     # [B, seq_len, num_modes * 2]
        uncertainty = self.uncertainty_layer(attn_output)      # [B, seq_len, num_modes * 2]

        # Reshape mu, b, uncertainty to [B, seq_len, num_modes, 2]
        B, T, _ = mu.shape
        mu = mu.view(B, T, self.num_modes, self.output_dim)
        b  = b.view(B, T, self.num_modes, self.output_dim)
        uncertainty = uncertainty.view(B, T, self.num_modes, self.output_dim)

        return pi, mu, b, uncertainty
