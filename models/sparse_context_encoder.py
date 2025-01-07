import torch
import torch.nn as nn
import pytorch_lightning as pl

class SparseContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SparseContextEncoder, self).__init__()
        self.agent_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.lane_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, agent_features, lane_features):
        agent_encoded, _ = self.agent_encoder(agent_features)  # (batch, seq_len, hidden_dim)
        lane_encoded, _ = self.lane_encoder(lane_features)
        combined = torch.cat([agent_encoded, lane_encoded], dim=-1)  # Concatenate features
        output = self.projection(combined)  # Project to final dimension
        return output


