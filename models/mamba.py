# models/mamba.py
import torch
import torch.nn as nn

class MambaBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MambaBlock, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        self.linear_out = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x

        x = self.linear_in(x)
        x = x.transpose(1, 2)  # (B, hidden_dim, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, seq_len, hidden_dim)
        x = self.silu(x)
        x = self.linear_out(x)

        x = x + residual
        x = self.layer_norm(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks):
        super(MambaLayer, self).__init__()
        self.blocks = nn.ModuleList([MambaBlock(input_dim, hidden_dim) for _ in range(num_blocks)])

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)
        return x
