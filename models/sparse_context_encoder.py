# models/sparse_context_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseContextEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=128, num_heads=4, num_layers=3):
        super(SparseContextEncoder, self).__init__()
        # Project agent features from input_dim to hidden_dim
        self.agent_projection = nn.Linear(input_dim, hidden_dim)
        #print(f"[SparseContextEncoder] agent_projection.weight.shape: {self.agent_projection.weight.shape}")

        # GRU to encode sequential agent features
        self.agent_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Self-Attention layers
        self.agent_self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

        # Final projection to output_dim
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, agent_features):
        """
        agent_features: [B, N=6, 4]
        """
        # Project input features
        agent_features = self.agent_projection(agent_features)  # [B, N, 128]
        #print(f"[SparseContextEncoder] after projection: {agent_features.shape}")

        # Encode with GRU
        agent_encoded, _ = self.agent_encoder(agent_features)  # [B, N, 128]
        #print(f"[SparseContextEncoder] after GRU: {agent_encoded.shape}")

        # Apply Self-Attention
        agent_self_attended = agent_encoded
        for i in range(self.num_layers):
            agent_self_attended, _ = self.agent_self_attention_layers[i](
                agent_self_attended, agent_self_attended, agent_self_attended
            )
            #print(f"[SparseContextEncoder] after attention {i}: {agent_self_attended.shape}")

        # Final projection
        output = self.projection(agent_self_attended)  # [B, N, 128]
        #print(f"[SparseContextEncoder] final output: {output.shape}")
        return output
