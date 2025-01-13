# models/lane_aware_probability_learning.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_layer import MambaLayer

class LaneAwareProbabilityLearning(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=128, num_lanes=6,
                 mamba_hidden_dim=128, num_mamba_blocks=1, dropout=0.1):
        super().__init__()
        self.num_lanes = num_lanes
        self.mamba_layer = MambaLayer(input_dim, mamba_hidden_dim, num_mamba_blocks)
        self.dropout = nn.Dropout(dropout)

        # Output layer: lane probabilities
        # Changed from 6 * num_lanes to num_lanes to align with cross-entropy
        self.linear = nn.Linear(mamba_hidden_dim, num_lanes)

    def forward(self, x):
        """
        x: [B, N+L, 3072]
        Returns:
            pi: [B, N+L, 6] (probabilities over 6 lanes)
            lane_preds: [B, N+L] (predicted lane indices)
        """
        # Pass through MambaLayer
        Fm = self.mamba_layer(x)  # [B, N+L, 128]
        Fm = self.dropout(Fm)

        # Compute lane probabilities
        out = self.linear(Fm)     # [B, N+L, 6]
        pi = F.softmax(out, dim=-1)  # [B, N+L, 6]

        # Predicted lane indices
        lane_preds = torch.argmax(pi, dim=-1)  # [B, N+L]

        return pi, lane_preds
