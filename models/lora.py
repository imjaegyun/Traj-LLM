# models/lora.py

import torch
import torch.nn as nn
import math

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        """
        x: [B, N, D]
        """
        out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling  # [B, N, D]
        return out
