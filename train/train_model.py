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
            hidden_dim=config.modules.high_level_model.hidden_dim,
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


    def forward(self, agent_inputs, lane_inputs):
        device = agent_inputs.device

        # Sparse context encoding
        sparse_features = self.sparse_encoder(agent_inputs, lane_inputs)

        # High-level interaction modeling
        high_level_features = self.high_level_model(sparse_features, device=device)

        # Lane-aware probability learning
        lane_probabilities, lane_predictions = self.lane_probability_model(high_level_features, lane_inputs)

        # Multimodal Laplace decoding
        mu, b, uncertainty = self.laplace_decoder(high_level_features)
        print(f"Mu shape: {mu.shape}, B shape: {b.shape}, Uncertainty shape: {uncertainty.shape}")

        return lane_probabilities, lane_predictions, mu, b




    def configure_optimizers(self):
        # AdamW Optimizer 설정
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.train.lr, weight_decay=self.config.train.weight_decay)
        return optimizer


    def training_step(self, batch, batch_idx):
        # 입력 데이터 추출
        agent_inputs = batch['agent_features']
        lane_inputs = batch['lane_features']
        trajectory_points = batch['trajectory_points']
        lane_labels = batch['lane_labels']  # [batch_size]

        # Forward Pass
        lane_probabilities, lane_predictions, mu, b = self(agent_inputs, lane_inputs)

        # `num_classes`를 동적으로 설정
        num_classes = lane_probabilities.shape[-1]  # 마지막 차원이 num_classes

        # Reshape `lane_probabilities`와 `lane_labels`
        lane_probabilities = lane_probabilities.reshape(-1, num_classes)  # [batch_size * seq_len, num_classes]
        lane_labels = lane_labels.reshape(-1)  # [batch_size * seq_len]

        # Loss 계산
        lane_loss = nn.CrossEntropyLoss()(lane_probabilities, lane_labels)
        laplace_loss = self.laplace_decoder.compute_laplace_loss(mu, b, trajectory_points)
        total_loss = lane_loss + laplace_loss

        # Wandb 로깅
        wandb.log({
            "train/total_loss": total_loss.item(),
            "train/lane_loss": lane_loss.item(),
            "train/laplace_loss": laplace_loss.item()
        })

        # Lightning 로깅
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_lane_loss", lane_loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_laplace_loss", laplace_loss, on_step=True, on_epoch=True, logger=True)

        return total_loss


    def validation_step(self, batch, batch_idx):
        agent_inputs = batch['agent_features']
        lane_inputs = batch['lane_features']
        lane_labels = batch['lane_labels']  # Shape: [batch_size, seq_len]

        # Forward Pass
        lane_probabilities, lane_predictions, mu, b = self(agent_inputs, lane_inputs)

        # Reshape lane_labels and lane_probabilities for CrossEntropyLoss
        batch_size, seq_len, num_classes = lane_probabilities.shape
        lane_probabilities = lane_probabilities.reshape(-1, num_classes)  # Shape: [batch_size * seq_len, num_classes]
        lane_labels = lane_labels.reshape(-1)  # Shape: [batch_size * seq_len]
        print(f"lane_probabilities shape: {lane_probabilities.shape}, lane_labels shape: {lane_labels.shape}")
        # Compute loss
        lane_loss = nn.CrossEntropyLoss()(lane_probabilities, lane_labels)

        laplace_loss = self.laplace_decoder.compute_laplace_loss(mu, b, batch["trajectory_points"])
        total_loss = lane_loss + laplace_loss

        # Log validation metrics
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, logger=True)
        return total_loss


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        agent_inputs = batch['agent_features']
        lane_inputs = batch['lane_features']

        try:
            # Forward Pass
            lane_probabilities, lane_predictions, mu, b = self(agent_inputs, lane_inputs)
            if lane_probabilities is None or lane_predictions is None:
                raise ValueError("Invalid output: forward pass returned None")

            return {
                "lane_probabilities": lane_probabilities.detach().cpu().numpy(),
                "lane_predictions": lane_predictions.detach().cpu().numpy(),
                "mu": mu.detach().cpu().numpy(),
                "b": b.detach().cpu().numpy()
            }
        except Exception as e:
            print(f"Error in predict_step for batch {batch_idx}: {e}")
            return {
                "lane_probabilities": None,
                "lane_predictions": None,
                "mu": None,
                "b": None,
                "error": str(e)
            }
    
def train_main(config: DictConfig):
    # Wandb Logger 초기화
    wandb_logger = WandbLogger(
        project=config.modules.wandb.project,
        entity=config.modules.wandb.get("entity", None),
        mode=config.modules.wandb.get("mode", "online")
    )

    # 모델 초기화
    model = TrajLLM(config)

    # Nuscenes DataLoader 설정
    nuscenes_path = config.modules.data.nuscenes_path
    train_dataset = NuscenesDataset(nuscenes_path=nuscenes_path, version="v1.0-trainval", split="train")
    val_dataset = NuscenesDataset(nuscenes_path=nuscenes_path, version="v1.0-trainval", split="val")

    # DataLoader 설정
    batch_size = config.train.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Trainer 초기화
    trainer = pl.Trainer(
        max_epochs=config.train.epochs,
        accelerator='gpu',  # GPU 가속 설정
        devices=config.train.gpus,  # 사용할 GPU 수
        logger=wandb_logger
    )

    # 학습 시작
    trainer.fit(model, train_loader, val_loader)


@hydra.main(config_path="/home/user/Traj-LLM/imjaegyun/Traj-LLM/configs", config_name="config.yaml")
def main(config: DictConfig):
    train_main(config)


if __name__ == "__main__":
    main()
