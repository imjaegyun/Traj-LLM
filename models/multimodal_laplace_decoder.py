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
        """
        Forward pass to compute \u03bc, b, \u03c0, and uncertainty.
        Args:
            high_level_features: Tensor of shape [batch_size, seq_len, input_dim]
            lane_probabilities: Tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            pi: Mixing coefficients of shape [batch_size, seq_len, num_modes]
            mu: Mean prediction of shape [batch_size, seq_len, num_modes, output_dim]
            b: Scale prediction of shape [batch_size, seq_len, num_modes, output_dim]
            uncertainty: Uncertainty prediction of shape [batch_size, seq_len, num_modes, output_dim]
        """
        # Cross Attention between high-level features and lane probabilities
        attn_output, _ = self.cross_attention(high_level_features, lane_probabilities, lane_probabilities)

        # Mixing coefficients (\u03c0)
        pi = F.softmax(self.mixing_layer(attn_output), dim=-1)  # Shape: [batch_size, seq_len, num_modes]

        # Mean (\u03bc) and scale (b)
        mu = self.mu_layer(attn_output).view(-1, attn_output.size(1), self.num_modes, self.output_dim)
        b = self.b_layer(attn_output).view(-1, attn_output.size(1), self.num_modes, self.output_dim)

        # Uncertainty prediction
        uncertainty = self.uncertainty_layer(attn_output).view(-1, attn_output.size(1), self.num_modes, self.output_dim)

        return pi, mu, b, uncertainty

    @staticmethod
    def compute_laplace_loss(pi, mu, b, targets):
        """
        Compute the combined Laplace loss using the Winner-Takes-All strategy.
        Args:
            pi: Mixing coefficients of shape [batch_size, seq_len, num_modes]
            mu: Predicted mean of shape [batch_size, seq_len, num_modes, output_dim]
            b: Predicted scale of shape [batch_size, seq_len, num_modes, output_dim]
            targets: Ground truth tensor of shape [batch_size, seq_len, output_dim]
        Returns:
            total_loss: Combined Laplace loss
        """
        epsilon = 1e-6  # To prevent division by zero
        b = b + epsilon

        # Expand targets for each mode
        targets = targets.unsqueeze(2).expand_as(mu)  # Shape: [batch_size, seq_len, num_modes, output_dim]

        # Compute Laplace negative log-likelihood for each mode
        diff = torch.abs(mu - targets)
        log_likelihood = torch.log(2 * b) + diff / b  # Shape: [batch_size, seq_len, num_modes, output_dim]

        # Sum over output_dim to get per-mode likelihood
        per_mode_loss = log_likelihood.sum(dim=-1)  # Shape: [batch_size, seq_len, num_modes]

        # Combine with mixing coefficients (\u03c0) using Winner-Takes-All
        weighted_loss = pi * per_mode_loss
        wta_loss, _ = torch.min(weighted_loss, dim=-1)  # Take the mode with minimum error

        total_loss = wta_loss.mean()  # Average over batch and sequence

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
