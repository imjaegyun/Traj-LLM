# models/multimodal_laplace_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalLaplaceDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_modes=5, num_lanes=6):
        super(MultimodalLaplaceDecoder, self).__init__()

        self.num_modes = num_modes
        self.output_dim = output_dim
        self.num_lanes = num_lanes
        self.input_dim = input_dim

        # (추가) lane probability -> input_dim 프로젝션
        # 예: 6차원 "lane_probabilities"를 128차원 embedding으로 만듦
        self.lane_prob_proj = nn.Linear(num_lanes, input_dim)

        # Mixing coefficients (π)
        self.mixing_layer = nn.Linear(input_dim, num_modes)

        # Mean (μ) prediction for each mode
        self.mu_layer = nn.Linear(input_dim, num_modes * output_dim)

        # Scale (b) prediction for each mode
        self.b_layer = nn.Sequential(
            nn.Linear(input_dim, num_modes * output_dim),
            nn.Softplus()  # Ensure positivity
        )

        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=4, 
            batch_first=True
        )

        # Final uncertainty prediction
        self.uncertainty_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes * output_dim)
        )

    def forward(self, high_level_features, lane_probabilities):
        """
        high_level_features: [B, seq_len, input_dim], e.g. [B, 6, 128]
        lane_probabilities : [B, seq_len, num_lanes=6]

        (주의) lane_probabilities가 6차원(클래스 수)이므로,
        그대로 MHA에 넣으면 'mat1 and mat2 shapes...' 에러 발생
        => 아래처럼 임베딩 차원으로 투영해서 사용
        """
        # 1) 6차원 -> input_dim(128) 투영
        lane_prob_emb = self.lane_prob_proj(lane_probabilities)  # [B, seq_len, 128]

        # 2) Cross-attention
        attn_output, _ = self.cross_attention(
            query=high_level_features,  # [B, seq_len, 128]
            key=lane_prob_emb,         # [B, seq_len, 128]
            value=lane_prob_emb        # [B, seq_len, 128]
        )
        #print(f"[DEBUG] Attention output shape: {attn_output.shape}")

        # 3) Compute pi (mixing coefficients)
        pi = F.softmax(self.mixing_layer(attn_output), dim=-1)
        #print(f"[DEBUG] Pi shape: {pi.shape}")

        # 4) Compute mu, b
        mu = self.mu_layer(attn_output).view(-1, attn_output.size(1), self.num_modes, self.output_dim)
        b = self.b_layer(attn_output).view(-1, attn_output.size(1), self.num_modes, self.output_dim)

        # 5) Uncertainty
        uncertainty = self.uncertainty_layer(attn_output).view(
            -1, attn_output.size(1), self.num_modes, self.output_dim
        )

        return pi, mu, b, uncertainty

    @staticmethod
    def compute_laplace_loss(pi, mu, b, targets):
        """
        Laplace negative log-likelihood
        """
        epsilon = 1e-6
        b = b + epsilon
        # targets: [B, seq_len, output_dim]
        targets = targets.unsqueeze(2).expand_as(mu)  # => [B, seq_len, num_modes, output_dim]

        diff = torch.abs(mu - targets)
        log_likelihood = torch.log(2 * b) + diff / b

        weighted_loss = pi * log_likelihood.sum(dim=-1)  # sum over output_dim
        wta_loss, _ = torch.min(weighted_loss, dim=-1)   # winner-takes-all
        total_loss = wta_loss.mean()
        return total_loss
