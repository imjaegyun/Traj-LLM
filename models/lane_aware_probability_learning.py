# models/lane_aware_probability_learning.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba import MambaLayer

class LaneAwareProbabilityLearning(nn.Module):
    def __init__(self, agent_dim, lane_dim, hidden_dim, num_lanes, num_mamba_blocks=3):
        super(LaneAwareProbabilityLearning, self).__init__()

        # Linear projections for input features
        self.linear_m = nn.Linear(agent_dim, hidden_dim)
        self.linear_n = nn.Linear(agent_dim, hidden_dim)
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

        # Structured state matrices
        self.state_matrix_a = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.state_matrix_b = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.state_matrix_c = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        # Learnable parameters for discretization
        self.delta_param = nn.Parameter(torch.randn(hidden_dim))

        # Mamba Layer
        self.mamba_layer = MambaLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, num_blocks=num_mamba_blocks)

        # Cross-Attention Layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Dropout and normalization
        self.dropout = nn.Dropout(0.1)
        self.instance_norm = nn.InstanceNorm1d(hidden_dim)

        # Linear for selective state-space model output
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)

        # Output layers
        self.mlp = nn.Linear(hidden_dim, num_lanes)

    def forward(self, high_level_features, lane_inputs):
        B, L, D = lane_inputs.size()

        # Linear projections and non-linearity
        m = self.linear_m(lane_inputs)  # Shape: (B, L, hidden_dim)
        n = self.linear_n(lane_inputs)  # Shape: (B, L, hidden_dim)
        print(f"[DEBUG] m shape: {m.shape}, values: {m[0, :2, :5]}")  # 첫 2개 행, 5개 열 출력
        print(f"[DEBUG] n shape: {n.shape}, values: {n[0, :2, :5]}")

        m_prime = F.silu(self.conv1d(m.transpose(1, 2))).transpose(1, 2)  # Shape: (B, L, hidden_dim)
        print(f"[DEBUG] m_prime shape: {m_prime.shape}, values: {m_prime[0, :2, :5]}")

        # Structured state-space model (SSM) matrices
        A = self.state_matrix_a
        B = self.state_matrix_b

        delta = F.softplus(self.delta_param).detach().unsqueeze(0).unsqueeze(0)
        print(f"[DEBUG] Delta shape: {delta.shape}, values: {delta[0, :5]}")  # delta 값 출력

        A_discrete = (A * delta).to(lane_inputs.device)
        B_discrete = (B * delta).to(lane_inputs.device)

        print(f"[DEBUG] A_discrete shape: {A_discrete.shape}, values: {A_discrete[:5, :5]}")
        print(f"[DEBUG] B_discrete shape: {B_discrete.shape}, values: {B_discrete[:5, :5]}")

        # Selective state-space model computations
        q_ssm = torch.bmm(A_discrete, m_prime.transpose(1, 2)) + torch.bmm(B_discrete, n.transpose(1, 2))
        q_ssm = q_ssm.transpose(1, 2)  # Shape: (B, L, hidden_dim)
        print(f"[DEBUG] q_ssm shape: {q_ssm.shape}, values: {q_ssm[0, :2, :5]}")

        # Process `q_ssm` through additional layers
        q = F.silu(self.linear_q(q_ssm))  # Shape: (B, L, hidden_dim)
        logits = self.mlp(q)  # Shape: (B, L, num_lanes)
        lane_probabilities = F.softmax(logits, dim=-1)  # Shape: (B, L, num_lanes)
        print(f"[DEBUG] logits shape: {logits.shape}, values: {logits[0, :2, :5]}")
        print(f"[DEBUG] lane_probabilities shape: {lane_probabilities.shape}, values: {lane_probabilities[0, :2, :5]}")

        # Lane predictions
        lane_predictions = torch.argmax(lane_probabilities, dim=-1)  # Shape: (B, L)
        print(f"[DEBUG] lane_predictions shape: {lane_predictions.shape}, values: {lane_predictions[0, :5]}")

        return lane_probabilities, lane_predictions
