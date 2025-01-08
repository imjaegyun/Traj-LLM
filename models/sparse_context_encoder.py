import torch
import torch.nn as nn

class SparseContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SparseContextEncoder, self).__init__()
        
        # Agent and Lane encoders
        self.agent_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.lane_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Fusion layers
        self.agent_lane_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lane_agent_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output projection
        self.projection = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, agent_features, lane_features):
        # Encode agent and lane features
        agent_encoded, _ = self.agent_encoder(agent_features)  # (batch, seq_len, hidden_dim)
        lane_encoded, _ = self.lane_encoder(lane_features)

        # Fusion: Agent-Lane (A-L)
        agent_lane_combined = torch.cat([agent_encoded, lane_encoded], dim=-1)
        agent_lane_fused = torch.relu(self.agent_lane_fusion(agent_lane_combined))

        # Fusion: Lane-Agent (L-A)
        lane_agent_combined = torch.cat([lane_encoded, agent_encoded], dim=-1)
        lane_agent_fused = torch.relu(self.lane_agent_fusion(lane_agent_combined))

        # Concatenate fused features
        fused_features = torch.cat([agent_lane_fused, lane_agent_fused], dim=-1)

        # Project to final output
        output = self.projection(fused_features)

        return output

# Example Usage
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    input_dim = 128
    hidden_dim = 256
    output_dim = 128

    encoder = SparseContextEncoder(input_dim, hidden_dim, output_dim)
    agent_features = torch.rand(batch_size, seq_len, input_dim)
    lane_features = torch.rand(batch_size, seq_len, input_dim)

    output = encoder(agent_features, lane_features)
    print(f"Output shape: {output.shape}")
