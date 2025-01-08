# models/high_level_interaction_model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class HighLevelInteractionModel(nn.Module):
    def __init__(self, llm_model_name, input_dim, hidden_dim, output_dim):
        super(HighLevelInteractionModel, self).__init__()

        # Load pre-trained LLM
        self.llm = AutoModel.from_pretrained(llm_model_name)

        # Input projection to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features, device):
        # Move features to the correct device
        features = features.to(device)

        # Project input features to hidden dimension
        projected_inputs = self.input_projection(features)

        # Attention mechanism
        attn_output, _ = self.attention(projected_inputs, projected_inputs, projected_inputs)

        # Feedforward processing
        output = self.feed_forward(attn_output)

        return output


# Example Usage
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    input_dim = 128
    hidden_dim = 256
    output_dim = 128
    llm_model_name = "meta-llama/Llama-3.2-3B"

    model = HighLevelInteractionModel(llm_model_name, input_dim, hidden_dim, output_dim)
    example_inputs = ["A car is turning left.", "A pedestrian is crossing.", "The vehicle is stopped at a red light."] * batch_size

    output = model(example_inputs)
    print(f"Output shape: {output.shape}")
