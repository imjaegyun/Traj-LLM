# models/mamba_layer.py

import torch
import torch.nn as nn

class MambaBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_m = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [B, N, D]
        out = self.linear_m(x)
        return self.activation(out)

class MambaLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock(input_dim, hidden_dim) for _ in range(num_blocks)])

    def forward(self, x):
        # x: [B, N, D]
        for block in self.blocks:
            x = block(x)
        return x
