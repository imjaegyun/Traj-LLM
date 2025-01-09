# models/multimodal_laplace_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalLaplaceDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_modes=5):
        super(MultimodalLaplaceDecoder, self).__init__()

        self.num_modes = num_modes
        self.output_dim = output_dim

        # Mixing coefficients (\u03c0)
        self.mixing_layer = nn.Linear(input_dim, num_modes)

        # Mean (\u03bc) prediction for each mode
        self.mu_layer = nn.Linear(input_dim, num_modes * output_dim)

        # Scale (b) prediction for each mode
        self.b_layer = nn.Sequential(
            nn.Linear(input_dim, num_modes * output_dim),
            nn.Softplus()  # Ensure positivity
        )

        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)

        # Final uncertainty prediction
        self.uncertainty_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes * output_dim)
        )

    def forward(self, high_level_features, lane_probabilities):
        attn_output, _ = self.cross_attention(high_level_features, lane_probabilities, lane_probabilities)
        print(f"[DEBUG] Attention output shape: {attn_output.shape}, values: {attn_output[0, :2, :5]}")

        pi = F.softmax(self.mixing_layer(attn_output), dim=-1)
        print(f"[DEBUG] Pi shape: {pi.shape}, values: {pi[0, :2, :5]}")

        mu = self.mu_layer(attn_output).view(-1, attn_output.size(1), self.num_modes, self.output_dim)
        b = self.b_layer(attn_output).view(-1, attn_output.size(1), self.num_modes, self.output_dim)
        print(f"[DEBUG] Mu shape: {mu.shape}, values: {mu[0, 0, :2, :2]}")
        print(f"[DEBUG] B shape: {b.shape}, values: {b[0, 0, :2, :2]}")

        uncertainty = self.uncertainty_layer(attn_output).view(-1, attn_output.size(1), self.num_modes, self.output_dim)
        print(f"[DEBUG] Uncertainty shape: {uncertainty.shape}, values: {uncertainty[0, 0, :2, :2]}")

        return pi, mu, b, uncertainty

    @staticmethod
    def compute_laplace_loss(pi, mu, b, targets):
        epsilon = 1e-6
        b = b + epsilon
        targets = targets.unsqueeze(2).expand_as(mu)

        diff = torch.abs(mu - targets)
        log_likelihood = torch.log(2 * b) + diff / b
        print(f"[DEBUG] Log-likelihood shape: {log_likelihood.shape}, values: {log_likelihood[0, 0, :2]}")

        weighted_loss = pi * log_likelihood.sum(dim=-1)
        print(f"[DEBUG] Weighted loss shape: {weighted_loss.shape}, values: {weighted_loss[0, :2]}")

        wta_loss, _ = torch.min(weighted_loss, dim=-1)
        print(f"[DEBUG] WTA loss shape: {wta_loss.shape}, values: {wta_loss[:2]}")

        total_loss = wta_loss.mean()
        return total_loss



# Example Usage
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    input_dim = 128
    output_dim = 2
    num_modes = 3

    model = MultimodalLaplaceDecoder(input_dim, output_dim, num_modes)

    # Example features and targets
    high_level_features = torch.rand(batch_size, seq_len, input_dim)
    lane_probabilities = torch.rand(batch_size, seq_len, input_dim)
    targets = torch.rand(batch_size, seq_len, output_dim)

    # Forward pass
    pi, mu, b, uncertainty = model(high_level_features, lane_probabilities)

    # Compute loss
    loss = model.compute_laplace_loss(pi, mu, b, targets)

    print(f"Pi Shape: {pi.shape}")
    print(f"Mu Shape: {mu.shape}")
    print(f"B Shape: {b.shape}")
    print(f"Uncertainty Shape: {uncertainty.shape}")
    print(f"Laplace Loss: {loss.item()}")
