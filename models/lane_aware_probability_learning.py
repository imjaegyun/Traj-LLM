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

        # Output layers
        self.mlp = nn.Linear(hidden_dim, num_lanes)

    def forward(self, high_level_features, lane_inputs):
        B, L, D = lane_inputs.size()

        # Linear projections and non-linearity
        m = self.linear_m(lane_inputs)  # Shape: (B, L, hidden_dim)
        n = self.linear_n(lane_inputs)  # Shape: (B, L, hidden_dim)
        m_prime = F.silu(self.conv1d(m.transpose(1, 2))).transpose(1, 2)  # Shape: (B, L, hidden_dim)

        # Pass through Mamba Layer
        q = self.mamba_layer(m_prime)  # Shape: (B, L, hidden_dim)

        # Structured state-space model (SSM) matrices
        A = self.state_matrix_a
        B = self.state_matrix_b
        C = self.state_matrix_c

        # Discretization of the matrices
        delta = F.softplus(self.delta_param).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, hidden_dim)
        A_discrete = A * delta
        B_discrete = B * delta

        # Selective state-space model computations
        q_ssm = torch.bmm(A_discrete.expand(B, -1, -1), q) + torch.bmm(B_discrete.expand(B, -1, -1), n)
        q_ssm = F.silu(q_ssm)

        # Cross-Attention
        cross_attended, _ = self.cross_attention(q_ssm, high_level_features, high_level_features)

        # Normalization and dropout
        q_final = self.instance_norm(self.dropout(cross_attended.transpose(1, 2))).transpose(1, 2)  # Shape: (B, L, hidden_dim)

        # Probability computation
        logits = self.mlp(q_final)  # Shape: (B, L, num_lanes)
        lane_probabilities = F.softmax(logits, dim=-1)  # Shape: (B, L, num_lanes)

        # Lane predictions
        lane_predictions = torch.argmax(lane_probabilities, dim=-1)  # Shape: (B, L)

        return lane_probabilities, lane_predictions

# Example Usage
if __name__ == "__main__":
    batch_size = 4
    num_lanes = 5
    seq_len = 10
    agent_dim = 128
    lane_dim = 128
    hidden_dim = 256
    num_mamba_blocks = 3

    model = LaneAwareProbabilityLearning(agent_dim, lane_dim, hidden_dim, num_lanes, num_mamba_blocks)
    high_level_features = torch.rand(batch_size, seq_len, hidden_dim)
    lane_inputs = torch.rand(batch_size, seq_len, lane_dim)

    lane_probabilities, lane_predictions = model(high_level_features, lane_inputs)
    print(f"Lane Probabilities Shape: {lane_probabilities.shape}")
    print(f"Lane Predictions Shape: {lane_predictions.shape}")