import torch
import torch.nn as nn
from transformers import LlamaModel

class LoRA(nn.Module):
    def __init__(self, input_dim, rank):
        super(LoRA, self).__init__()
        self.A = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, input_dim) * 0.01)

    def forward(self, W, x):
        return torch.matmul(W, x) + torch.matmul(self.B, torch.matmul(self.A, x))

class HighLevelInteractionModel(nn.Module):
    def __init__(self, llm_model_name, input_dim, output_dim, use_lora=False, lora_rank=4):
        super(HighLevelInteractionModel, self).__init__()

        # Load pre-trained Llama model
        self.llm = LlamaModel.from_pretrained(llm_model_name)

        # Freeze all parameters in the Llama model
        for param in self.llm.parameters():
            param.requires_grad = False

        # Extract Llama's hidden size
        llama_hidden_size = self.llm.config.hidden_size

        # Input projection to match Llama's hidden size
        self.input_projection = nn.Linear(input_dim, llama_hidden_size)

        # Optional LoRA for input projection
        self.use_lora = use_lora
        if self.use_lora:
            self.lora_input = LoRA(input_dim=llama_hidden_size, rank=lora_rank)

        # Scaled Dot-Product Attention with LoRA
        self.query_projection = nn.Linear(llama_hidden_size, llama_hidden_size)
        self.key_projection = nn.Linear(llama_hidden_size, llama_hidden_size)
        self.value_projection = nn.Linear(llama_hidden_size, llama_hidden_size)

        if self.use_lora:
            self.lora_query = LoRA(input_dim=llama_hidden_size, rank=lora_rank)
            self.lora_key = LoRA(input_dim=llama_hidden_size, rank=lora_rank)
            self.lora_value = LoRA(input_dim=llama_hidden_size, rank=lora_rank)

        self.attention_output_projection = nn.Linear(llama_hidden_size, llama_hidden_size)

        # Add & Layer Norm after Attention and Feedforward
        self.attention_norm = nn.LayerNorm(llama_hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(llama_hidden_size, 4 * llama_hidden_size),
            nn.ReLU(),
            nn.Linear(4 * llama_hidden_size, llama_hidden_size),
        )
        self.feed_forward_norm = nn.LayerNorm(llama_hidden_size)

        # Output projection to match required output_dim
        self.output_projection = nn.Linear(llama_hidden_size, output_dim)

    def forward(self, features, device):
        """
        Forward pass for high-level interaction modeling.
        Args:
            features (torch.Tensor): [batch_size, seq_len, input_dim]
            device (torch.device): The device to use
        Returns:
            torch.Tensor: [batch_size, seq_len, output_dim]
        """
        # Move features to the correct device
        features = features.to(device)

        # Project input features to Llama's hidden size
        projected_inputs = self.input_projection(features)
        #print("[DEBUG] Projected inputs shape:", projected_inputs.shape)

        if self.use_lora:
            projected_inputs = self.lora_input(self.input_projection.weight, projected_inputs)
            #print("[DEBUG] LoRA applied to inputs:", projected_inputs.shape)

        # Llama forward pass
        llama_outputs = self.llm(inputs_embeds=projected_inputs)
        hidden_states = llama_outputs.last_hidden_state
        #print("[DEBUG] Llama hidden states shape:", hidden_states.shape)

        # Scaled Dot-Product Attention with LoRA
        queries = self.query_projection(hidden_states)
        keys = self.key_projection(hidden_states)
        values = self.value_projection(hidden_states)
        #print("[DEBUG] Queries shape:", queries.shape)
        #print("[DEBUG] Keys shape:", keys.shape)
        #print("[DEBUG] Values shape:", values.shape)

        if self.use_lora:
            queries = self.lora_query(self.query_projection.weight, queries)
            keys = self.lora_key(self.key_projection.weight, keys)
            values = self.lora_value(self.value_projection.weight, values)
            #print("[DEBUG] LoRA applied to Queries/Keys/Values")

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (keys.size(-1) ** 0.5)
        #print("[DEBUG] Attention scores shape:", attention_scores.shape)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_output = torch.matmul(attention_probs, values)
        #print("[DEBUG] Attention output shape:", attention_output.shape)

        # Attention output projection
        attention_output = self.attention_output_projection(attention_output)
        #print("[DEBUG] Projected attention output shape:", attention_output.shape)

        # Add & Norm after Attention
        attention_output = self.attention_norm(attention_output + hidden_states)

        # Feedforward network
        feed_forward_output = self.feed_forward(attention_output)
        #print("[DEBUG] Feedforward output shape:", feed_forward_output.shape)

        # Add & Norm after Feedforward
        output = self.feed_forward_norm(feed_forward_output + attention_output)

        # Project to final output dimension
        output = self.output_projection(output)
        #print("[DEBUG] Final output shape:", output.shape)

        return output
