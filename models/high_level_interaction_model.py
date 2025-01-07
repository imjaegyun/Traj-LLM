import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer

class HighLevelInteractionModel(pl.LightningModule):
    def __init__(self, llm_model_name, input_dim, hidden_dim, output_dim):
        super(HighLevelInteractionModel, self).__init__()

        # Pre-trained LLM
        self.llm = AutoModel.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        # Social Dot-product Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # MLP for final feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, inputs):
        # Tokenize inputs for LLM
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        # Pass inputs through Pre-trained LLM
        llm_outputs = self.llm(**tokenized_inputs).last_hidden_state

        # Apply Social Dot-product Attention
        attention_output, _ = self.attention(llm_outputs, llm_outputs, llm_outputs)

        # Add & Normalize
        attention_output = self.layer_norm(attention_output + llm_outputs)

        # Pass through MLP
        output = self.mlp(attention_output[:, -1, :])  # Take the last token representation

        return output

# Example usage
if __name__ == "__main__":
    # Example input dimensions
    batch_size = 8
    seq_len = 10
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
    model_name = "meta-llama/Llama-3.2"  # Replace with Llama 3.2 model name on Hugging Face
    model = HighLevelInteractionModel(model_name, input_dim, hidden_dim, output_dim)

    # Forward pass
    outputs = model(inputs)
    print("Output shape:", outputs.shape)
