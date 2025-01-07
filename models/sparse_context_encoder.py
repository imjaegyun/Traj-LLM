import torch
import torch.nn as nn
import pytorch_lightning as pl

class SparseContextEncoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SparseContextEncoder, self).__init__()

        # GRU for Agent-Agent, Agent-Lane, and Lane-Agent Encoding
        self.agent_agent_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.agent_lane_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.lane_agent_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # MLP for feature fusion in each interaction
        self.agent_agent_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.agent_lane_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.lane_agent_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Final Fusion Layer
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, agent_inputs, lane_inputs):
        # Validate input dimensions
        assert agent_inputs.size(1) == lane_inputs.size(1), "Sequence lengths of agent_inputs and lane_inputs must match."
        
        # Agent-Agent encoding
        agent_agent_outputs, _ = self.agent_agent_gru(agent_inputs)
        agent_agent_features = self.agent_agent_mlp(agent_agent_outputs[:, -1, :])

        # Agent-Lane encoding
        agent_lane_outputs, _ = self.agent_lane_gru(agent_inputs)
        agent_lane_features = self.agent_lane_mlp(agent_lane_outputs[:, -1, :])

        # Lane-Agent encoding
        lane_agent_outputs, _ = self.lane_agent_gru(lane_inputs)
        lane_agent_features = self.lane_agent_mlp(lane_agent_outputs[:, -1, :])

        # Feature fusion
        combined_features = torch.cat([agent_agent_features, agent_lane_features, lane_agent_features], dim=-1)
        fused_features = self.fusion_mlp(combined_features)

        return fused_features

# Example usage
if __name__ == "__main__":
    # Example input dimensions
    batch_size = 8
    seq_len = 10
    input_dim = 16
    hidden_dim = 32
    output_dim = 64

    # Random example inputs
    agent_inputs = torch.randn(batch_size, seq_len, input_dim)
    lane_inputs = torch.randn(batch_size, seq_len, input_dim)

    # Model initialization
    model = SparseContextEncoder(input_dim, hidden_dim, output_dim)

    # Forward pass
    outputs = model(agent_inputs, lane_inputs)
    print("Output shape:", outputs.shape)
