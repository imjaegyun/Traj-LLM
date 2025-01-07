# models/high_level_interaction_model.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

class HighLevelInteractionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HighLevelInteractionModel, self).__init__()

        # LLM-like Transformer
        self.transformer_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=2
        )

        # Input Projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output Projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        # Project inputs
        inputs = self.input_projection(inputs)

        # Pass through Transformer
        transformer_output = self.transformer_layer(inputs)

        # Layer Normalization
        normalized_output = self.layer_norm(transformer_output)

        # Output Projection
        output = self.output_projection(normalized_output)

        return output


# Example usage
if __name__ == "__main__":
    # Example input dimensions
    batch_size = 8
    input_dim = 16
    hidden_dim = 64
    output_dim = 128

    # Example text inputs (batch of sentences)
    inputs = [
        "The car is driving on the highway.",
        "A pedestrian is crossing the street.",
        "There is a traffic jam on the main road."
    ] * (batch_size // 3)

    # Model initialization
    model_name = "meta-llama/Llama-3.2-3B"  # Replace with your model name on Hugging Face
    model = HighLevelInteractionModel(model_name, input_dim, hidden_dim, output_dim)

    # Forward pass
    outputs = model(inputs)
    print("Output shape:", outputs.shape)
