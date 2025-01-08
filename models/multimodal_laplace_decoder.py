# models/multimodal_laplace_decoder.py
import torch
import torch.nn as nn

class MultimodalLaplaceDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultimodalLaplaceDecoder, self).__init__()

        # Mean (mu) prediction
        self.mu_layer = nn.Linear(input_dim, output_dim)

        # Scale (b) prediction
        self.b_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softplus()
        )

        # Laplace uncertainty estimation layers
        self.laplace_uncertainty_layer = nn.Sequential(
            nn.Linear(input_dim, 64),  # Hidden layer size
            nn.ReLU(),
            nn.Linear(64, output_dim)  # Output matches mu and b
        )

    def forward(self, features):
        """
        Forward pass to compute mu, b, and uncertainty.
        Args:
            features: Input tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            mu: Mean prediction of shape [batch_size, seq_len, output_dim]
            b: Scale prediction of shape [batch_size, seq_len, output_dim]
            uncertainty: Uncertainty prediction of shape [batch_size, seq_len, output_dim]
        """
        # Ensure features have expected shape
        print(f"Input features shape: {features.shape}")

        mu = self.mu_layer(features)
        b = self.b_layer(features)
        uncertainty = self.laplace_uncertainty_layer(features)

        # Debugging shapes
        print(f"Mu shape: {mu.shape}, B shape: {b.shape}, Uncertainty shape: {uncertainty.shape}")

        return mu, b, uncertainty

    @staticmethod
    def compute_laplace_loss(mu, b, targets):
        """
        Compute Laplace negative log-likelihood loss.
        Args:
            mu: Predicted mean of shape [batch_size, seq_len, output_dim]
            b: Predicted scale of shape [batch_size, seq_len, output_dim]
            targets: Ground truth
             targets: Ground truth tensor of shape [batch_size, seq_len, output_dim]
        Returns:
            loss: Computed Laplace loss as a scalar tensor
        """
        epsilon = 1e-6  # To prevent division by zero
        b = b + epsilon

        # Compute Laplace negative log-likelihood
        diff = torch.abs(mu - targets)
        log_likelihood = torch.log(2 * b) + diff / b
        loss = log_likelihood.mean()  # Average over all elements

        # Debugging the computed loss
        print(f"Computed Laplace Loss: {loss.item()}")

        return loss


# Example Usage
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    input_dim = 128
    output_dim = 64

    model = MultimodalLaplaceDecoder(input_dim, output_dim)

    # Example features and targets
    features = torch.rand(batch_size, seq_len, input_dim)
    targets = torch.rand(batch_size, seq_len, output_dim)

    # Forward pass
    mu, b, uncertainty = model(features)

    # Compute loss
    loss = model.compute_laplace_loss(mu, b, targets)

    print(f"Mu (Mean) Shape: {mu.shape}")
    print(f"B (Scale) Shape: {b.shape}")
    print(f"Uncertainty Shape: {uncertainty.shape}")
    print(f"Laplace Loss: {loss.item()}")
