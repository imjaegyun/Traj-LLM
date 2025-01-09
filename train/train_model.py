# train/train_model.py
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
from data.nuscenes_data_loader import NuscenesDataset
import hydra
from omegaconf import DictConfig
from models.mamba import MambaLayer

class TrajLLM(pl.LightningModule):
    def __init__(self, config):
        super(TrajLLM, self).__init__()

        # Configuration
        self.config = config
        self.k_values = config.train.k_values
        self.learning_rate = config.train.lr

        # Sparse Context Encoding
        self.sparse_encoder = SparseContextEncoder(
            input_dim=config.modules.sparse_encoder.input_dim,
            hidden_dim=config.modules.sparse_encoder.hidden_dim,
            output_dim=config.modules.sparse_encoder.output_dim
        )

        # High-level Interaction Modeling
        if not hasattr(config.modules.high_level_model, "llm_model_name"):
            raise ValueError("llm_model_name is missing in config.modules.high_level_model")

        self.high_level_model = HighLevelInteractionModel(
            llm_model_name=config.modules.high_level_model.llm_model_name,
            input_dim=config.modules.high_level_model.input_dim,
            output_dim=config.modules.high_level_model.output_dim
        )

        # Lane-aware Probability Learning
        self.lane_probability_model = LaneAwareProbabilityLearning(
            agent_dim=config.modules.lane_probability.agent_dim,
            lane_dim=config.modules.lane_probability.lane_dim,
            hidden_dim=config.modules.lane_probability.hidden_dim,
            num_lanes=config.modules.lane_probability.num_lanes
        )

        # Multimodal Laplace Decoder
        self.laplace_decoder = MultimodalLaplaceDecoder(
            input_dim=config.modules.laplace_decoder.input_dim,
            output_dim=config.modules.laplace_decoder.output_dim
        )

        # Mamba Layer (Optional)
        self.mamba_layer = MambaLayer(
            input_dim=config.modules.mamba.input_dim,
            hidden_dim=config.modules.mamba.hidden_dim,
            num_blocks=config.modules.mamba.num_blocks
        ) if hasattr(config.modules, "mamba") else None

        # init weights
        self.reset_parameters()
        torch.set_float32_matmul_precision("medium")

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, agent_inputs, lane_inputs):
        device = agent_inputs.device

        # 1) Sparse context encoding
        sparse_features = self.sparse_encoder(agent_inputs, lane_inputs)
        # shape: (B, 6, <encoder_output_dim>)

        # 2) High-level interaction
        high_level_features = self.high_level_model(sparse_features, device=device)
        # shape: (B, 6, <high_level_model_output_dim>)

        # 3) Lane-aware probability learning
        lane_probabilities, lane_predictions = self.lane_probability_model(high_level_features, lane_inputs)
        # lane_probabilities: (B, seq_len, num_lanes), lane_predictions: (B, seq_len)

        # (Optional) Mamba layer
        if self.mamba_layer:
            high_level_features = self.mamba_layer(high_level_features)

        # 4) Multimodal Laplace decoding
        pi, mu, b, uncertainty = self.laplace_decoder(high_level_features, lane_probabilities)

        return lane_probabilities, lane_predictions, pi, mu, b, uncertainty

    def compute_minade(self, predictions, ground_truth):
        # predictions, ground_truth: (..., 2)
        return torch.mean(torch.norm(predictions - ground_truth, dim=-1))

    def compute_minfde(self, predictions, ground_truth):
        # 마지막 프레임 위치 차이
        return torch.norm(predictions[:, -1] - ground_truth[:, -1], dim=-1).mean()

    def compute_miss_rate(self, predictions, ground_truth, threshold=2.0):
        # 마지막 프레임 오차가 threshold 초과되는 비율
        final_displacement = torch.norm(predictions[:, -1] - ground_truth[:, -1], dim=-1)
        return (final_displacement > threshold).float().mean()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.train.lr_step,
            gamma=self.config.train.lr_gamma
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """
        여기서 실제 Loss를 계산.
        가정:
          - batch['lane_labels']: (B, seq_len)  차선 GT
          - batch['trajectory_points']: (B, T, 2)  궤적 GT
        """
        agent_inputs = batch['agent_features']     
        lane_inputs = batch['lane_features']       
        lane_labels = batch.get('lane_labels', None)  # 정답 차선 인덱스
        traj_points = batch.get('trajectory_points', None)

        lane_probs, lane_preds, pi, mu, b, uncertainty = self(agent_inputs, lane_inputs)

        # 1) Lane Loss (CrossEntropy 예시)
        lane_loss = torch.tensor(0.0, device=self.device)
        if lane_labels is not None:
            # lane_probs: (B, L, num_lanes), lane_labels: (B, L)
            B, L, num_lanes = lane_probs.shape
            lane_loss_fn = nn.CrossEntropyLoss()
            lane_loss = lane_loss_fn(
                lane_probs.view(B*L, num_lanes),
                lane_labels.view(B*L)
            )

        # 2) Trajectory Loss (Laplace 예시)
        laplace_loss = torch.tensor(0.0, device=self.device)
        if traj_points is not None and mu is not None:
            # pi, mu, b: (B, 6, num_modes, 2)  (가정)
            # traj_points: (B, 240, 2) -> 예: 6프레임만 사용하려면 slicing 필요
            # 예: 6프레임만 자르기
            gt_6 = traj_points[:, :6, :]  # (B, 6, 2) assuming we only want the first 6 frames to match mu.shape
            laplace_loss = self.laplace_decoder.compute_laplace_loss(pi, mu, b, gt_6)

        # 최종 손실
        total_loss = lane_loss + laplace_loss

        # 로깅
        self.log("train/lane_loss", lane_loss, on_step=True, on_epoch=True)
        self.log("train/laplace_loss", laplace_loss, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        agent_inputs = batch['agent_features']
        lane_inputs = batch['lane_features']
        traj_points = batch.get('trajectory_points', None)
        lane_labels = batch.get('lane_labels', None)

        lane_probs, lane_preds, pi, mu, b, uncertainty = self(agent_inputs, lane_inputs)

        # (옵션) Lane Loss
        lane_loss = torch.tensor(0.0, device=self.device)
        if lane_labels is not None:
            B, L, num_lanes = lane_probs.shape
            lane_loss = nn.CrossEntropyLoss()(
                lane_probs.view(B*L, num_lanes),
                lane_labels.view(B*L)
            )

        # (옵션) Laplace Loss
        laplace_loss = torch.tensor(0.0, device=self.device)
        if traj_points is not None and mu is not None:
            # 6프레임만 사용 (가정)
            gt_6 = traj_points[:, :6, :]
            laplace_loss = self.laplace_decoder.compute_laplace_loss(pi, mu, b, gt_6)

        total_loss = lane_loss + laplace_loss
        self.log("val/lane_loss", lane_loss, on_epoch=True)
        self.log("val/laplace_loss", laplace_loss, on_epoch=True)
        self.log("val/total_loss", total_loss, on_epoch=True)

        # minADE / minFDE / miss_rate 계산 (첫 모드만 사용 예시)
        if mu is not None and mu.ndim == 4:
            # mu: (B, 6, num_modes, 2)
            pred = mu[:, :, 0, :]  # (B, 6, 2) 첫 모드
            if traj_points is not None:
                gt_6 = traj_points[:, :6, :]
                minADE = self.compute_minade(pred, gt_6)
                minFDE = self.compute_minfde(pred, gt_6)
                miss_rate = self.compute_miss_rate(pred, gt_6)
            else:
                minADE = torch.tensor(0.0, device=self.device)
                minFDE = torch.tensor(0.0, device=self.device)
                miss_rate = torch.tensor(0.0, device=self.device)

            self.log("val_minADE", minADE)
            self.log("val_minFDE", minFDE)
            self.log("val_miss_rate", miss_rate)

        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        agent_inputs = batch['agent_features']
        lane_inputs = batch['lane_features']
        lane_probs, lane_preds, pi, mu, b, uncertainty = self(agent_inputs, lane_inputs)

        return {
            "lane_probabilities": lane_probs.detach().cpu().numpy(),
            "lane_predictions": lane_preds.detach().cpu().numpy(),
            "pi": pi.detach().cpu().numpy(),
            "mu": mu.detach().cpu().numpy(),
            "b": b.detach().cpu().numpy(),
            "uncertainty": uncertainty.detach().cpu().numpy()
        }

def train_main(config: DictConfig):
    wandb_logger = WandbLogger(
        project=config.modules.wandb.project,
        entity=config.modules.wandb.get("entity", None),
        mode=config.modules.wandb.get("mode", "online")
    )

    model = TrajLLM(config)

    nuscenes_path = config.modules.data.nuscenes_path
    train_dataset = NuscenesDataset(nuscenes_path=nuscenes_path, version="v1.0-trainval", split="train")
    val_dataset = NuscenesDataset(nuscenes_path=nuscenes_path, version="v1.0-trainval", split="val")

    batch_size = config.train.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    trainer = pl.Trainer(
        max_epochs=config.train.epochs,
        accelerator='gpu',
        devices=config.train.gpus,
        logger=wandb_logger,
        gradient_clip_val=config.train.gradient_clip_val
    )

    trainer.fit(model, train_loader, val_loader)


@hydra.main(version_base="1.1",
            config_path="/home/user/Traj-LLM/imjaegyun/Traj-LLM/configs",
            config_name="config.yaml")
def main(config: DictConfig):
    train_main(config)

if __name__ == "__main__":
    main()
