# train/train_model.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models.sparse_context_encoder import SparseContextEncoder
from models.high_level_interaction_model import HighLevelInteractionModel
from models.lane_aware_probability_learning import LaneAwareProbabilityLearning
from models.multimodal_laplace_decoder import MultimodalLaplaceDecoder
from data.nuscenes_data_loader import NuscenesDatasetFiltered
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

class TrajLLM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = config.train.lr

        # Sparse Encoder
        self.sparse_encoder = SparseContextEncoder(
            input_dim=config.modules.sparse_encoder.input_dim,
            hidden_dim=config.modules.sparse_encoder.hidden_dim,
            output_dim=config.modules.sparse_encoder.output_dim,
            num_heads=config.modules.fusion.num_heads,
            num_layers=3
        )

        # High-level Model
        self.high_level_model = HighLevelInteractionModel(
            llm_model_name=config.modules.high_level_model.llm_model_name,
            input_dim=config.modules.sparse_encoder.output_dim,
            output_dim=config.modules.high_level_model.output_dim,
            use_lora=config.modules.high_level_model.use_lora,
            lora_rank=config.modules.high_level_model.lora_rank,
            num_heads=config.modules.fusion.num_heads
        )

        # Lane Probability
        self.lane_aware_probability = LaneAwareProbabilityLearning(
            input_dim=config.modules.lane_aware_probability.input_dim,
            hidden_dim=config.modules.lane_aware_probability.hidden_dim,
            num_lanes=config.modules.lane_aware_probability.num_lanes,
            mamba_hidden_dim=128,
            num_mamba_blocks=1,
            dropout=0.1
        )

        # Laplace Decoder
        self.laplace_decoder = MultimodalLaplaceDecoder(
            input_dim=config.modules.laplace_decoder.input_dim,  # Should be 6
            output_dim=config.modules.laplace_decoder.output_dim,
            num_modes=config.modules.laplace_decoder.num_modes
        )

        # Loss Functions
        self.laplace_loss_fn = self.compute_laplace_loss
        self.ce_loss_fn = nn.CrossEntropyLoss()

        self.validation_outputs = []
        torch.set_float32_matmul_precision("medium")

    def forward(self, agent_inputs, lane_inputs):
        """
        agent_inputs: [B, N, 4]
        lane_inputs : [B, L, 4]
        """
        # 1) Sparse Encoder => [B, N, 128]
        sparse_features = self.sparse_encoder(agent_inputs)

        # 2) High-level Model => [B, N+L, 3072]
        device = sparse_features.device
        high_level_features = self.high_level_model(sparse_features, device)

        # 3) Lane Probability => pi_lane [B, N+L, 6]
        pi_lane, lane_preds = self.lane_aware_probability(high_level_features)

        # 4) Laplace Decoder => [pi_laplace, mu, b, uncertainty]
        pi_laplace, mu, b, uncertainty = self.laplace_decoder(high_level_features, pi_lane)

        return pi_laplace, mu, b, uncertainty, pi_lane

    def compute_laplace_loss(self, pi, mu, b, traj_points):
        """
        Compute the Weighted Negative Log-Likelihood loss for the Laplace distribution.
        """
        epsilon = 1e-6
        b = b + epsilon  # Ensure numerical stability

        # Select the last timestep
        pi_last = pi[:, -1, :]          # [B, num_modes]
        mu_last = mu[:, -1, :, :]       # [B, num_modes, 2]
        b_last  = b[:, -1, :, :]        # [B, num_modes, 2]

        # Ground truth for the last timestep
        final_gt = traj_points[:, -1, :]    # [B, 2]
        targets = final_gt.unsqueeze(1).expand(-1, mu_last.size(1), -1)  # [B, num_modes, 2]

        # Compute the Laplace log-likelihood
        diff = torch.abs(mu_last - targets)
        ll = torch.log(2 * b_last) + diff / b_last  # [B, num_modes, 2]

        # Sum over the output dimensions (e.g., x and y)
        ll_sum = ll.sum(dim=-1)  # [B, num_modes]

        # Weighted loss using pi_last
        weighted_loss = pi_last * ll_sum  # [B, num_modes]

        # Compute the expectation instead of WTA
        total_loss = weighted_loss.mean()

        return total_loss

    def _get_lane_labels(self, batch):
        """
        Extract lane labels from the batch.
        This function needs to be implemented based on how lane labels are stored in your dataset.
        """
        lane_labels = batch.get('lane_labels', None)
        if lane_labels is None:
            raise ValueError("Lane labels are not available in the batch.")
        return lane_labels

    def training_step(self, batch, batch_idx):
        if batch is None:
            return torch.tensor(0.0, device=self.device)

        agent_in = batch['agent_features']  # [B, N, 4]
        lane_in  = batch['lane_features']   # [B, L, 4]
        traj_pts = batch['trajectory_points']  # [B, target_length, 2]

        # Forward pass
        pi_laplace, mu, b, uncertainty, pi_lane = self(agent_in, lane_in)  # pi_lane: [B, N+L, 6]

        # Compute lane labels
        try:
            lane_labels = self._get_lane_labels(batch)  # [B, L]
        except ValueError as e:
            logger.error(e)
            return torch.tensor(0.0, device=self.device)

        # Compute losses
        laplace_loss = self.laplace_loss_fn(pi_laplace, mu, b, traj_pts)
        
        # Extract lane-related probabilities
        pi_lane_lanes = pi_lane[:, -self.config.modules.lane_aware_probability.num_lanes:, :]  # [B, L, 6]

        # Compute cross-entropy loss using reshape instead of view
        lane_loss = self.ce_loss_fn(
            pi_lane_lanes.reshape(-1, self.config.modules.lane_aware_probability.num_lanes),  # [B*L, 6]
            lane_labels.reshape(-1)  # [B*L]
        )

        # Combine losses
        total_loss = laplace_loss + self.config.train.lambda_lane * lane_loss

        # Check for NaNs or Infs
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"[Training Step] NaN/Inf in total_loss at batch_idx={batch_idx}")
            return torch.tensor(0.0, device=self.device)

        # Log losses separately
        self.log("train/loss_step", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/laplace_loss", laplace_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/lane_loss", lane_loss, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss


    def validation_step(self, batch, batch_idx):
        if batch is None:
            return

        # Input data
        agent_in = batch['agent_features']  # [B, N, 4]
        lane_in = batch['lane_features']    # [B, L, 4]
        traj_pts = batch['trajectory_points']  # [B, target_length, 2]

        # Forward pass
        pi_laplace, mu, b, uncertainty, pi_lane = self(agent_in, lane_in)

        # Debug shapes
        print(f"mu.shape: {mu.shape}, traj_pts.shape: {traj_pts.shape}")

        # Ensure mu's dimensions match expectations
        batch_size, num_modes, T, spatial_dim = mu.shape  # [B, num_modes, T, 2]
        _, target_length, _ = traj_pts.shape  # [B, target_length, 2]

        # If time steps do not match, downsample traj_pts
        if T != target_length:
            print(f"Aligning traj_pts from {target_length} to {T} time steps.")
            traj_pts = traj_pts[:, :T, :]  # Take the first T time steps

        # Expand traj_pts to match num_modes
        gt_traj = traj_pts.unsqueeze(1).expand(-1, num_modes, -1, -1)  # [B, num_modes, T, 2]

        # Compute L2 distance (norm)
        diff = torch.norm(mu - gt_traj, dim=-1)  # [B, num_modes, T]
        minADE = diff.mean(dim=-1).min(dim=-1).values.mean().item()  # [B] -> scalar
        minFDE = diff[:, :, -1].min(dim=-1).values.mean().item()  # [B] -> scalar

        # Compute Miss Rate (MR)
        threshold = 2.0  # Example: 2 meters
        miss = (diff[:, :, -1].min(dim=-1).values > threshold).float()  # [B]
        MR = miss.mean().item()

        # Log metrics
        self.log("val/minADE", minADE, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/minFDE", minFDE, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/MR", MR, on_step=False, on_epoch=True, prog_bar=True)

        # Compute lane labels and losses
        try:
            lane_labels = self._get_lane_labels(batch)  # [B, L]
        except ValueError as e:
            logger.error(e)
            return

        # Compute Laplace and lane loss
        laplace_loss = self.laplace_loss_fn(pi_laplace, mu, b, traj_pts)
        pi_lane_lanes = pi_lane[:, -self.config.modules.lane_aware_probability.num_lanes:, :]  # [B, L, 6]

        lane_loss = self.ce_loss_fn(
            pi_lane_lanes.reshape(-1, self.config.modules.lane_aware_probability.num_lanes),  # [B*L, 6]
            lane_labels.reshape(-1)  # [B*L]
        )

        # Total loss
        total_loss = laplace_loss + self.config.train.lambda_lane * lane_loss
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss




    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.train.lr_gamma,
            patience=3,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    @staticmethod
    def collate_fn(batch):
        filtered_batch = [s for s in batch if s is not None]
        if not filtered_batch:
            return None
        agent_feats = torch.stack([s['agent_features'] for s in filtered_batch])  # [B, N, 4]
        lane_feats  = torch.stack([s['lane_features']  for s in filtered_batch])  # [B, L, 4]
        traj_pts    = torch.stack([s['trajectory_points'] for s in filtered_batch])  # [B, target_length, 2]
        lane_labels = torch.stack([s.get('lane_labels', torch.zeros(1, dtype=torch.long)) for s in filtered_batch])  # [B, L]
        return {
            'agent_features': agent_feats,
            'lane_features': lane_feats,
            'trajectory_points': traj_pts,
            'lane_labels': lane_labels
        }

@hydra.main(version_base=None, config_path="config.yaml")
def train_main(config: DictConfig):
    logger.info("[train_main] Initializing TrajLLM model...")
    logger.info("Final Config:")
    logger.info(OmegaConf.to_yaml(config, sort_keys=False))

    wandb_logger = WandbLogger(
        project=config.wandb.project,
        mode=config.wandb.get("mode", "online")
    )

    model = TrajLLM(config)

    ds_train = NuscenesDatasetFiltered(
        nuscenes_path=config.data.nuscenes_path,
        version="v1.0-trainval",
        split=config.data.train_split,
        target_length=config.modules.data_loader.target_length,
        num_agents=config.modules.data_loader.num_agents,
        num_lanes=config.modules.data_loader.num_lanes
    )
    ds_val = NuscenesDatasetFiltered(
        nuscenes_path=config.data.nuscenes_path,
        version="v1.0-trainval",
        split=config.data.val_split,
        target_length=config.modules.data_loader.target_length,
        num_agents=config.modules.data_loader.num_agents,
        num_lanes=config.modules.data_loader.num_lanes
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=TrajLLM.collate_fn
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=TrajLLM.collate_fn
    )

    trainer = pl.Trainer(
        max_epochs=config.train.epochs,
        logger=wandb_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config.train.gpus if torch.cuda.is_available() else None,
        gradient_clip_val=config.train.gradient_clip_val,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=10
    )

    logger.info("[train_main] Starting trainer.fit()...")
    trainer.fit(model, loader_train, loader_val)

if __name__ == "__main__":
    train_main()
