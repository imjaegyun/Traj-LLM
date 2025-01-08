# models/lane_aware_probability_learning.py
import torch
import torch.nn as nn
from models.mamba import MambaLayer

class LaneAwareProbabilityLearning(nn.Module):
    def __init__(self, agent_dim, lane_dim, hidden_dim, num_lanes):
        super(LaneAwareProbabilityLearning, self).__init__()
        
        # Mamba Layer for lane feature enhancement
        self.mamba_layer = MambaLayer(lane_dim, hidden_dim, num_blocks=4)

        # Cross Attention between agent and lane features
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Lane probability prediction layers
        self.linear_in = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, num_lanes)

    def forward(self, agent_features, lane_features):
        try:
            print(f"Agent features shape: {agent_features.shape}")
            print(f"Lane features shape: {lane_features.shape}")

            # Mamba Layer
            enhanced_lane_features = self.mamba_layer(lane_features)
            print(f"Enhanced lane features shape: {enhanced_lane_features.shape}")

            # Cross Attention
            attn_output, _ = self.cross_attention(enhanced_lane_features, agent_features, agent_features)
            print(f"Cross-attended features shape: {attn_output.shape}")

            # Lane Probability
            lane_probabilities = self.linear_out(self.linear_in(attn_output))
            print(f"Lane probabilities shape: {lane_probabilities.shape}")

            lane_predictions = torch.argmax(lane_probabilities, dim=-1)
            print(f"Lane predictions shape: {lane_predictions.shape}")

            return lane_probabilities, lane_predictions
        except Exception as e:
            print(f"Error in LaneAwareProbabilityLearning: {e}")
            return None, None

# Example Usage
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    agent_dim = 128
    lane_dim = 128
    hidden_dim = 256
    num_lanes = 5

    model = LaneAwareProbabilityLearning(agent_dim, lane_dim, hidden_dim, num_lanes)
    agent_features = torch.rand(batch_size, seq_len, agent_dim)
    lane_features = torch.rand(batch_size, seq_len, lane_dim)

    lane_probabilities, lane_predictions = model(agent_features, lane_features)
    print("Lane Probabilities:", lane_probabilities)
    print("Lane Predictions:", lane_predictions)
