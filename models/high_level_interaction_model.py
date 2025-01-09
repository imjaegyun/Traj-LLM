# models/high_level_interaction_model.py
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import LlamaModel

class LoRA(nn.Module):
    def __init__(self, input_dim, rank):
        super(LoRA, self).__init__()
        self.A = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, input_dim) * 0.01)

    def forward(self, W, x):
        return torch.matmul(W, x) + torch.matmul(self.B, torch.matmul(self.A, x))

class HighLevelInteractionModel(nn.Module):
    def __init__(self, llm_model_name, input_dim, output_dim):
        super(HighLevelInteractionModel, self).__init__()

        # Load pre-trained Llama model
        self.llm = LlamaModel.from_pretrained(llm_model_name)

        # Extract Llama's hidden size
        llama_hidden_size = self.llm.config.hidden_size

        # Input projection to match Llama's hidden size
        self.input_projection = nn.Linear(input_dim, llama_hidden_size)

        # Output projection to match required output_dim
        self.output_projection = nn.Linear(llama_hidden_size, output_dim)

    def forward(self, features, device):
        """
        Forward pass for high-level interaction modeling.
        Args:
            features (torch.Tensor): Input tensor [batch_size, seq_len, input_dim]
            device (torch.device): The device to use (e.g., cuda)
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, output_dim]
        """
        # Move features to the correct device
        features = features.to(device)
        print(f"[DEBUG] Input features shape: {features.shape}, device: {features.device}")

        # Project input features to Llama's hidden size
        projected_inputs = self.input_projection(features)
        print(f"[DEBUG] Projected inputs shape: {projected_inputs.shape}")

        # Llama forward pass
        llama_outputs = self.llm(inputs_embeds=projected_inputs)
        hidden_states = llama_outputs.last_hidden_state
        print(f"[DEBUG] Llama hidden states shape: {hidden_states.shape}")

        # Project to final output dimension
        output = self.output_projection(hidden_states)
        print(f"[DEBUG] Final output shape: {output.shape}")

        return output



# Example Usage
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    input_dim = 128
    hidden_dim = 256
    output_dim = 128
    llm_model_name = "gpt2"

    model = HighLevelInteractionModel(llm_model_name, input_dim, hidden_dim, output_dim)
    inputs = torch.rand(batch_size, seq_len, input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output = model(inputs, device)
    print(f"Output shape: {output.shape}")
