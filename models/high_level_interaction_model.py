# models/high_level_interaction_model.py

import torch
import torch.nn as nn
from transformers import LlamaModel
from models.lora import LoRA
from models.fusion_submodule import FusionSubmodule

class HighLevelInteractionModel(nn.Module):
    def __init__(self, llm_model_name, input_dim, output_dim,
                 use_lora=False, lora_rank=4, num_heads=8):
        super().__init__()
        self.llm = LlamaModel.from_pretrained(llm_model_name)

        # LLaMA 파라미터 동결
        for param in self.llm.parameters():
            param.requires_grad = False

        llm_hidden_size = self.llm.config.hidden_size  # ex) 3072 for a 3B Llama?
        self.input_projection = nn.Linear(input_dim, llm_hidden_size)
        self.fusion = FusionSubmodule(hidden_dim=llm_hidden_size, num_heads=num_heads)
        self.use_lora = use_lora

        # LoRA 적용
        if self.use_lora:
            for layer in self.llm.layers:
                original_q_proj = layer.self_attn.q_proj
                layer.self_attn.q_proj = nn.Sequential(
                    original_q_proj,
                    LoRA(in_features=original_q_proj.out_features,
                         out_features=original_q_proj.out_features,
                         rank=lora_rank, alpha=1.0, dropout=0.0)
                )
                original_v_proj = layer.self_attn.v_proj
                layer.self_attn.v_proj = nn.Sequential(
                    original_v_proj,
                    LoRA(in_features=original_v_proj.out_features,
                         out_features=original_v_proj.out_features,
                         rank=lora_rank, alpha=1.0, dropout=0.0)
                )

        self.output_projection = nn.Linear(llm_hidden_size, output_dim)

    def forward(self, gi, device):
        """
        gi: [B, N, input_dim=128]
        => after projection => [B, N, llm_hidden=3072?]
        => fusion => LLM => out => [B, N, 3072]
        => output_projection => [B, N, output_dim=3072]
        """
        #print(f"[HighLevelInteractionModel] input gi: {gi.shape}")
        gi = gi.to(device)

        projected_inputs = self.input_projection(gi)
        #print(f"[HighLevelInteractionModel] after input_projection: {projected_inputs.shape}")

        fused_features = self.fusion(projected_inputs, projected_inputs)
        #print(f"[HighLevelInteractionModel] after fusion: {fused_features.shape}")

        llm_outputs = self.llm(inputs_embeds=fused_features)
        hidden_states = llm_outputs.last_hidden_state
        #print(f"[HighLevelInteractionModel] after LLM: {hidden_states.shape}")

        output = self.output_projection(hidden_states)
        #print(f"[HighLevelInteractionModel] after output_projection: {output.shape}")

        return output  # e.g. [B,N,3072]
