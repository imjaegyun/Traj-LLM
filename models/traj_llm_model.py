import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from models.sparse_context_encoder import SparseContextEncoder
from models.high_level_interaction_model import HighLevelInteractionModel
from models.lane_aware_probability_learning import LaneAwareProbabilityLearning
from models.multimodal_laplace_decoder import MultimodalLaplaceDecoder

class TrajLLM(pl.LightningModule):
    def __init__(self, config):
        super(TrajLLM, self).__init__()

        # Configuration
        self.config = config

        # Sparse Context Encoding
        self.sparse_encoder = SparseContextEncoder(
            input_dim=config.sparse_encoder.input_dim,
            hidden_dim=config.sparse_encoder.hidden_dim,
            output_dim=config.sparse_encoder.output_dim
        )

        # High-level Interaction Modeling
        self.high_level_model = HighLevelInteractionModel(
            llm_model_name=config.high_level_model.llm_model_name,
            input_dim=config.high_level_model.input_dim,
            hidden_dim=config.high_level_model.hidden_dim,
            output_dim=config.high_level_model.output_dim
        )

        # Lane-aware Probability Learning
        self.lane_probability_model = LaneAwareProbabilityLearning(
            agent_dim=config.lane_probability.agent_dim,
            lane_dim=config.lane_probability.lane_dim,
            hidden_dim=config.lane_probability.hidden_dim,
            num_lanes=config.lane_probability.num_lanes
        )

        # Multimodal Laplace Decoder
        self.laplace_decoder = MultimodalLaplaceDecoder(
            input_dim=config.laplace_decoder.input_dim,
            output_dim=config.laplace_decoder.output_dim
        )

    def forward(self, agent_inputs, lane_inputs):
        # Step 1: Sparse Context Joint Encoding
        sparse_features = self.sparse_encoder(agent_inputs, lane_inputs)

        # Step 2: High-level Interaction Modeling
        high_level_features = self.high_level_model(sparse_features)

        # Step 3: Lane-aware Probability Learning
        lane_probabilities, lane_predictions = self.lane_probability_model(high_level_features, lane_inputs)

        # Step 4: Multimodal Laplace Decoding
        mu, b = self.laplace_decoder(high_level_features)

        return lane_probabilities, lane_predictions, mu, b

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.train.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        agent_inputs, lane_inputs, targets = batch

        # Forward pass
        lane_probabilities, lane_predictions, mu, b = self(agent_inputs, lane_inputs)

        # Compute Loss
        lane_loss = nn.CrossEntropyLoss()(lane_probabilities, targets["lane_labels"])
        laplace_loss = self.laplace_decoder.compute_laplace_loss(mu, b, targets["trajectory_points"])
        total_loss = lane_loss + laplace_loss

        # Log to Wandb
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, logger=True)
        self.log("lane_loss", lane_loss, on_step=True, on_epoch=True, logger=True)
        self.log("laplace_loss", laplace_loss, on_step=True, on_epoch=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        agent_inputs, lane_inputs, targets = batch

        # Forward pass
        lane_probabilities, lane_predictions, mu, b = self(agent_inputs, lane_inputs)

        # Compute Loss
        lane_loss = nn.CrossEntropyLoss()(lane_probabilities, targets["lane_labels"])
        laplace_loss = self.laplace_decoder.compute_laplace_loss(mu, b, targets["trajectory_points"])
        total_loss = lane_loss + laplace_loss

        # Log to Wandb
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, logger=True)
        self.log("val_lane_loss", lane_loss, on_step=False, on_epoch=True, logger=True)
        self.log("val_laplace_loss", laplace_loss, on_step=False, on_epoch=True, logger=True)

        return total_loss

# Example usage with Hydra and Wandb
if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="configs", config_name="traj_llm")
    def main(config: DictConfig):
        # Initialize Wandb Logger
        wandb_logger = WandbLogger(project="Traj-LLM", config=config)

        # Model initialization
        model = TrajLLM(config)

        # DataLoader placeholder (replace with actual dataloader)
        train_loader = None
        val_loader = None

        # Trainer initialization
        trainer = pl.Trainer(
            max_epochs=config.train.epochs,
            gpus=config.train.gpus,
            logger=wandb_logger
        )

        # Start training
        trainer.fit(model, train_loader, val_loader)

    main()
