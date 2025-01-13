# models/fusion_submodule.py

import torch
import torch.nn as nn

class MultiSelfAtt(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: [B, S, H]
        attn_output, _ = self.multihead_attn(x, x, x)
        #print(f"[MultiSelfAtt] input={x.shape} => output={attn_output.shape}")
        return attn_output

class MultiCrossAtt(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, query, key):
        # query: [B, S, H], key: [B, L, H]
        attn_output, _ = self.multihead_attn(query, key, key)
        #print(f"[MultiCrossAtt] query={query.shape}, key={key.shape} => output={attn_output.shape}")
        return attn_output

class FusionSubmodule(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.multi_self_attn = MultiSelfAtt(hidden_dim, num_heads)
        self.glu = nn.GLU(dim=-1)
        self.multi_cross_attn_lane_agent = MultiCrossAtt(hidden_dim, num_heads)
        self.multi_cross_attn_agent_lane = MultiCrossAtt(hidden_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, agent_features, lane_features):
        """
        agent_features: [B, N, H]
        lane_features: [B, L, H]
        """
        # Agent-Agent Self-Attention
        hei = self.multi_self_attn(agent_features)  # [B, N, H]
        cat_hei = torch.cat([agent_features, hei], dim=-1)  # [B, N, 2H]
        hei = self.glu(cat_hei)  # => [B, N, H]
        hei = self.layer_norm1(hei + agent_features)
        #print(f"[FusionSubmodule] agent_features after GLU+LN={hei.shape}")

        # Lane-Agent Cross Attention
        cross_la = self.multi_cross_attn_lane_agent(lane_features, hei)  # [B, L, H]
        fel = lane_features + cross_la
        fel = self.layer_norm2(fel)
        #print(f"[FusionSubmodule] lane_features after cross_attn+LN={fel.shape}")

        # Agent-Lane Cross Attention
        cross_al = self.multi_cross_attn_agent_lane(hei, fel)  # [B, N, H]
        hei = hei + cross_al
        #print(f"[FusionSubmodule] agent_features after cross_attn={hei.shape}")

        # Concatenate Agent and Lane Features
        gi = torch.cat([hei, fel], dim=1)  # [B, N+L, H]
        #print(f"[FusionSubmodule] gi final={gi.shape}")
        return gi
