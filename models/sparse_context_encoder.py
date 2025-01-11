# models/sparse_context_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=3):
        super(SparseContextEncoder, self).__init__()

        # agent는 (B, 6, 4) -> (B, 6, input_dim)
        self.agent_projection = nn.Linear(4, input_dim)

        # lane은 이미 (B, 240, 128) -> (B, 240, 128)라고 가정
        self.lane_projection = nn.Identity()

        # GRU encoders for agent and lane features
        self.agent_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.lane_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Multi-layer Self-Attention for agent
        self.agent_self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(num_layers)
        ])

        # Multi-layer Cross-Attention for agent-lane and lane-agent
        self.agent_lane_cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(num_layers)
        ])
        self.lane_agent_cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(num_layers)
        ])

        # GLU for selective feature fusion
        self.agent_glu = nn.GLU(dim=-1)
        self.lane_glu = nn.GLU(dim=-1)

        # Final projection
        self.projection = nn.Linear(hidden_dim * 2, output_dim)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, agent_features, lane_features):
        """
        agent_features: (B, 6, 4)
        lane_features : (B, 240, 128)
        """
        # 1) Projection
        agent_features = self.agent_projection(agent_features)  # (B, 6, input_dim)
        lane_features = self.lane_projection(lane_features)     # (B, 240, 128) 그대로

        # 2) GRU encoding
        agent_encoded, _ = self.agent_encoder(agent_features)   # (B, 6, hidden_dim)
        lane_encoded, _ = self.lane_encoder(lane_features)      # (B, 240, hidden_dim)

        #print(f"[DEBUG] Agent encoded shape: {agent_encoded.shape}")
        #print(f"[DEBUG] Lane encoded shape: {lane_encoded.shape}")

        # 임시 변수로 저장
        agent_self_attended = agent_encoded
        # CrossAttention 결과(Agent->Lane, Lane->Agent)
        agent_lane_attended = lane_encoded
        lane_agent_attended = agent_encoded

        # 3) Multi-layer attentions
        for i in range(self.num_layers):
            # 3-1) Agent Self-Attention
            agent_self_attended, _ = self.agent_self_attention_layers[i](
                agent_self_attended, agent_self_attended, agent_self_attended
            )
            #print(f"[DEBUG] Layer {i} - Agent self-attended shape: {agent_self_attended.shape}")

            # 3-2) Agent-Lane Cross-Attention
            #     쿼리: Lane => 출력 [B, 240, hidden_dim]
            agent_lane_attended, _ = self.agent_lane_cross_attention_layers[i](
                lane_encoded,   # query
                agent_self_attended,  # key
                agent_self_attended   # value
            )
            #print(f"[DEBUG] Layer {i} - Agent-Lane attended shape: {agent_lane_attended.shape}")

            # 3-3) Lane-Agent Cross-Attention
            #     쿼리: Agent => 출력 [B, 6, hidden_dim]
            lane_agent_attended, _ = self.lane_agent_cross_attention_layers[i](
                agent_self_attended,  # query
                lane_encoded,         # key
                lane_encoded          # value
            )
            #print(f"[DEBUG] Layer {i} - Lane-Agent attended shape: {lane_agent_attended.shape}")

            # (중요) agent_lane_attended는 [B, 240, hidden_dim]
            #       agent_self_attended는 [B,   6, hidden_dim]
            # cat하려면 시퀀스 길이가 달라 에러 발생.
            # => lane 쪽(240)을 mean-pooling으로 압축 후, agent 길이(6)에 맞춤
            pooled_lane = agent_lane_attended.mean(dim=1, keepdim=True)   # [B, 1, hidden_dim]
            pooled_lane = pooled_lane.expand(-1, agent_self_attended.size(1), -1)  # [B, 6, hidden_dim]
            agent_lane_attended = pooled_lane

        # 이제 agent_self_attended와 agent_lane_attended 둘 다 [B, 6, hidden_dim]
        fused_agent = self.agent_glu(torch.cat([agent_self_attended, agent_lane_attended], dim=-1))
        #print(f"[DEBUG] Fused agent shape: {fused_agent.shape}")

        # lane_encoded: [B, 240, hidden_dim]
        # lane_agent_attended: [B, 6, hidden_dim]
        # 이 둘도 길이가 다르므로 pooling해서 맞춤
        pooled_lane_encoded = lane_encoded.mean(dim=1, keepdim=True)      # [B, 1, hidden_dim]
        pooled_lane_encoded = pooled_lane_encoded.expand(-1, lane_agent_attended.size(1), -1)
        # => [B, 6, hidden_dim]

        fused_lane = self.lane_glu(torch.cat([pooled_lane_encoded, lane_agent_attended], dim=-1))
        #print(f"[DEBUG] Fused lane shape: {fused_lane.shape}")

        # 최종 (B, 6, 2*hidden_dim)
        fused_features = torch.cat([fused_agent, fused_lane], dim=-1)

        output = self.projection(fused_features)  # (B, 6, output_dim)
        #print(f"[DEBUG] Final output shape: {output.shape}")

        return output