# models/lane_aware_probability_learning.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.mamba import MambaLayer

class LaneAwareProbabilityLearning(nn.Module):
    def __init__(self, agent_dim, lane_dim, hidden_dim, num_lanes):
        super(LaneAwareProbabilityLearning, self).__init__()
        assert hidden_dim == 128, f"Expected hidden_dim=128, but got {hidden_dim}"  # 추가 디버깅
        self.mamba_layer = MambaLayer(lane_dim, hidden_dim, num_blocks=4)
        self.linear_in = nn.Linear(hidden_dim, hidden_dim)  # Ensure correct hidden_dim
        self.linear_out = nn.Linear(hidden_dim, num_lanes)

    def forward(self, agent_features, lane_features):
        print(f"Lane features before MambaLayer: {lane_features.shape}")
        enhanced_lane_features = self.mamba_layer(lane_features)
        print(f"Lane features after MambaLayer: {enhanced_lane_features.shape}")

        batch_size, seq_length, hidden_dim = enhanced_lane_features.shape
        assert hidden_dim == 128, f"Expected hidden_dim=128, but got {hidden_dim}"  # 추가 디버깅
        enhanced_lane_features = enhanced_lane_features.reshape(-1, hidden_dim)
        print(f"Lane features after flattening: {enhanced_lane_features.shape}")

        enhanced_lane_features = self.linear_in(enhanced_lane_features)
        print(f"Lane features after linear_in: {enhanced_lane_features.shape}")

        enhanced_lane_features = enhanced_lane_features.reshape(batch_size, seq_length, -1)
        print(f"Lane features reshaped back: {enhanced_lane_features.shape}")

        lane_probabilities = self.linear_out(enhanced_lane_features)
        print(f"Lane probabilities shape: {lane_probabilities.shape}")

        return lane_probabilities, None





