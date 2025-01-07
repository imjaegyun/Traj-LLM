import torch
import torch.nn as nn
import pytorch_lightning as pl

class MultimodalLaplaceDecoder(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(MultimodalLaplaceDecoder, self).__init__()

        # Mean (mu) prediction
        self.mu_layer = nn.Linear(input_dim, output_dim)

        # Scale (b) prediction - must be positive, so we use softplus
        self.b_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softplus()
        )

    def forward(self, features):
        # Predict the mean (mu) and scale (b)
        mu = self.mu_layer(features)
        b = self.b_layer(features)

        return mu, b

    def compute_laplace_loss(self, mu, b, targets):
        # Laplace Negative Log-Likelihood Loss
        error = torch.abs(mu - targets)
        loss = torch.mean(torch.log(2 * b) + error / b)
        return loss

# Example usage
if __name__ == "__main__":
    # Example input dimensions
    batch_size = 8
    input_dim = 256  # Example input features from previous layers
    output_dim = 2   # Predict x, y coordinates

    # Random example inputs
    features = torch.randn(batch_size, input_dim)
    targets = torch.randn(batch_size, output_dim)  # Ground truth trajectory points

    # Model initialization
    decoder = MultimodalLaplaceDecoder(input_dim, output_dim)

    # Forward pass
    mu, b = decoder(features)

    # Compute loss
    loss = decoder.compute_laplace_loss(mu, b, targets)

    print("Predicted mean (mu):", mu)
    print("Predicted scale (b):", b)
    print("Laplace loss:", loss.item())
