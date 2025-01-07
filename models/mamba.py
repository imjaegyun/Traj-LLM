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
        self.layer_norm = nn.LayerNorm(input_dim)  # Consistent with input_dim

    def forward(self, x):
        # Debugging: Input shape
        print(f"Input shape before processing: {x.shape}")

        # Save residual
        residual = x

        # Linear layer transformation
        x = self.linear_in(x)
        print(f"Shape after linear_in: {x.shape}")

        # Transpose for Conv1D
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_length)
        x = self.conv(x)
        print(f"Shape after Conv1D: {x.shape}")

        # Transpose back and apply LayerNorm
        x = x.transpose(1, 2)  # Back to (batch_size, seq_length, hidden_dim)
        x = self.silu(x)
        x = self.linear_out(x)
        print(f"Shape after linear_out: {x.shape}")

        # Add residual and normalize
        x = x + residual
        x = self.layer_norm(x)
        print(f"Output shape after residual and LayerNorm: {x.shape}")
        return x

class MambaLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks):
        super(MambaLayer, self).__init__()
        self.blocks = nn.ModuleList([MambaBlock(input_dim, hidden_dim) for _ in range(num_blocks)])

    def forward(self, x):
        print(f"Input to MambaLayer: {x.shape}")
        for idx, block in enumerate(self.blocks):
            print(f"Passing through MambaBlock {idx}")
            x = block(x)
            print(f"Output shape after MambaBlock {idx}: {x.shape}")
        return x
