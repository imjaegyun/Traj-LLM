import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=3):
        super(SparseContextEncoder, self).__init__()

        # Input projection to match GRU input_dim
        self.agent_projection = nn.Linear(4, input_dim)  # 4차원을 128차원으로 변환
        self.lane_projection = nn.Linear(4, input_dim)   # 4차원을 128차원으로 변환

        # GRU encoders for agent and lane features
        self.agent_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.lane_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Multi-layer Self-Attention for agent features
        self.agent_self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(num_layers)
        ])

        # Multi-layer Cross-Attention for agent-lane and lane-agent fusion
        self.agent_lane_cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(num_layers)
        ])
        self.lane_agent_cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(num_layers)
        ])

        # Gated Linear Units (GLU) for selective feature fusion
        self.agent_glu = nn.GLU(dim=-1)
        self.lane_glu = nn.GLU(dim=-1)

        # Final projection layer
        self.projection = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, agent_features, lane_features):
        # Project input features to match GRU input_dim
        agent_features = self.agent_projection(agent_features)
        lane_features = self.lane_projection(lane_features)

        # Encode agent and lane features using GRU
        agent_encoded, _ = self.agent_encoder(agent_features)
        lane_encoded, _ = self.lane_encoder(lane_features)

        # 디버깅 메시지
        print(f"[DEBUG] Agent encoded shape: {agent_encoded.shape}")
        print(f"[DEBUG] Lane encoded shape: {lane_encoded.shape}")

        # Initialize attention outputs
        agent_self_attended = agent_encoded
        agent_lane_attended = lane_encoded
        lane_agent_attended = agent_encoded

        # Apply multiple layers of attention
        for i in range(self.num_layers):
            # Self-attention for agents
            agent_self_attended, _ = self.agent_self_attention_layers[i](
                agent_self_attended, agent_self_attended, agent_self_attended
            )
            print(f"[DEBUG] Layer {i} - Agent self-attended shape: {agent_self_attended.shape}")

            # Cross-attention: Agent-Lane (A-L)
            agent_lane_attended, _ = self.agent_lane_cross_attention_layers[i](
                lane_encoded, agent_self_attended, agent_self_attended
            )
            print(f"[DEBUG] Layer {i} - Agent-Lane attended shape: {agent_lane_attended.shape}")

            # Cross-attention: Lane-Agent (L-A)
            lane_agent_attended, _ = self.lane_agent_cross_attention_layers[i](
                agent_self_attended, lane_encoded, lane_encoded
            )
            print(f"[DEBUG] Layer {i} - Lane-Agent attended shape: {lane_agent_attended.shape}")

        # Fusion: Agent-Lane (A-L)
        fused_agent = self.agent_glu(torch.cat([agent_self_attended, agent_lane_attended], dim=-1))
        print(f"[DEBUG] Fused agent shape: {fused_agent.shape}")

        # Fusion: Lane-Agent (L-A)
        fused_lane = self.lane_glu(torch.cat([lane_encoded, lane_agent_attended], dim=-1))
        print(f"[DEBUG] Fused lane shape: {fused_lane.shape}")

        # Concatenate fused features from agent and lane
        fused_features = torch.cat([fused_agent, fused_lane], dim=-1)

        # Project to final output
        output = self.projection(fused_features)
        print(f"[DEBUG] Final output shape: {output.shape}")

        return output

# Example Usage
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    input_dim = 4  # Assume raw input has 4 dimensions (e.g., x, y, vx, vy)
    hidden_dim = 256
    output_dim = 128
    num_heads = 4
    num_layers = 3

    encoder = SparseContextEncoder(input_dim=128, hidden_dim=hidden_dim, output_dim=output_dim, num_heads=num_heads, num_layers=num_layers)
    agent_features = torch.rand(batch_size, seq_len, input_dim)
    lane_features = torch.rand(batch_size, seq_len, input_dim)

    output = encoder(agent_features, lane_features)
    print(f"Output shape: {output.shape}")
