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

        # MAMBA
        self.mamba_layer = MambaLayer(
            input_dim=config.modules.sparse_encoder.output_dim,
            hidden_dim=config.modules.mamba.hidden_dim,
            num_blocks=config.modules.mamba.num_blocks
        )

        # High-level Interaction Modeling
        self.high_level_model = HighLevelInteractionModel(
            input_dim=config.modules.high_level_model.input_dim,
            hidden_dim=config.modules.high_level_model.hidden_dim,
            output_dim=config.modules.high_level_model.output_dim
        )

        # Lane-aware Probability Learning
        print(f"Initializing LaneAwareProbabilityLearning with hidden_dim={config.modules.lane_probability.hidden_dim}")
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
        # Sparse Encoding
        sparse_features = self.sparse_encoder(agent_inputs, lane_inputs)
        print(f"Sparse features shape: {sparse_features.shape}")

        # MAMBA Layer
        mamba_features = self.mamba_layer(sparse_features)
        print(f"Mamba features shape: {mamba_features.shape}")

        # High-level Interaction
        high_level_features = self.high_level_model(mamba_features)
        print(f"High-level features shape: {high_level_features.shape}")

        # Lane Probability Learning
        lane_probabilities, lane_predictions = self.lane_probability_model(high_level_features, lane_inputs)
        print(f"Lane probabilities shape: {lane_probabilities.shape}")

        # Multimodal Laplace Decoding
        mu, b = self.laplace_decoder(high_level_features)
        print(f"Mu shape: {mu.shape}, B shape: {b.shape}")

        return lane_probabilities, lane_predictions, mu, b
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.train.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # 입력 데이터 추출
        agent_inputs = batch['agent_features']
        lane_inputs = batch['lane_features']
        trajectory_points = batch['trajectory_points']
        lane_labels = batch['lane_labels']  # [batch_size]

        # Forward Pass
        lane_probabilities, lane_predictions, mu, b = self(agent_inputs, lane_inputs)

        # 크기 확인 및 출력
        print(f"Lane probabilities shape: {lane_probabilities.shape}")  # [batch_size, num_lanes, num_classes]
        print(f"Lane labels shape: {lane_labels.shape}")  # [batch_size]

        # CrossEntropyLoss를 위해 올바른 크기로 변환
        # lane_probabilities에서 가장 높은 확률의 lane을 비교하기 위해 [batch_size, num_classes] 형태로 축소
        lane_probabilities = lane_probabilities.mean(dim=1)  # [batch_size, num_classes]

        # Loss 계산
        lane_loss = nn.CrossEntropyLoss()(lane_probabilities, lane_labels)
        laplace_loss = self.laplace_decoder.compute_laplace_loss(mu, b, trajectory_points)
        total_loss = lane_loss + laplace_loss

        # 로그 기록
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_lane_loss", lane_loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_laplace_loss", laplace_loss, on_step=True, on_epoch=True, logger=True)

        return total_loss


    def validation_step(self, batch, batch_idx):
        # 입력 데이터 추출
        agent_inputs = batch['agent_features']
        lane_inputs = batch['lane_features']
        trajectory_points = batch['trajectory_points']
        lane_labels = batch['lane_labels']  # [batch_size]

        # Forward Pass
        lane_probabilities, lane_predictions, mu, b = self(agent_inputs, lane_inputs)

        # 크기 확인 및 출력
        print(f"Lane probabilities shape: {lane_probabilities.shape}")  # [batch_size, num_lanes, num_classes]
        print(f"Lane labels shape: {lane_labels.shape}")  # [batch_size]

        # CrossEntropyLoss를 위해 크기 변환
        batch_size, num_lanes, num_classes = lane_probabilities.shape

        # lane_labels 확장: [batch_size] -> [batch_size, num_lanes]
        lane_labels = lane_labels.unsqueeze(1).expand(-1, num_lanes).reshape(-1)
        lane_probabilities = lane_probabilities.reshape(-1, num_classes)

        # 변환 후 크기 출력
        print(f"Reshaped lane probabilities shape: {lane_probabilities.shape}")  # [batch_size * num_lanes, num_classes]
        print(f"Reshaped lane labels shape: {lane_labels.shape}")  # [batch_size * num_lanes]

        # Loss 계산
        lane_loss = nn.CrossEntropyLoss()(lane_probabilities, lane_labels)
        laplace_loss = self.laplace_decoder.compute_laplace_loss(mu, b, trajectory_points)
        total_loss = lane_loss + laplace_loss

        # Wandb에 로깅
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, logger=True)
        self.log("val_lane_loss", lane_loss, on_step=False, on_epoch=True, logger=True)
        self.log("val_laplace_loss", laplace_loss, on_step=False, on_epoch=True, logger=True)

        return total_loss




def train_main(config: DictConfig):
    # Initialize Wandb Logger
    wandb_logger = WandbLogger(
        project=config.modules.wandb.project,
        entity=config.modules.wandb.get("entity", None),
        mode=config.modules.wandb.get("mode", "online")
    )

    # Model initialization
    model = TrajLLM(config)

    # Nuscenes DataLoader setup
    nuscenes_path = config.modules.data.nuscenes_path
    train_dataset = NuscenesDataset(nuscenes_path=nuscenes_path, version="v1.0-trainval", split="train")
    val_dataset = NuscenesDataset(nuscenes_path=nuscenes_path, version="v1.0-trainval", split="val")

    # Ensure `train` settings are loaded correctly
    batch_size = config.train.batch_size
    num_epochs = config.train.epochs
    learning_rate = config.train.lr
    gpus = config.train.gpus

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Trainer initialization
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu',  # GPU 가속 설정
        devices=gpus,       # 사용할 GPU 수
        logger=wandb_logger
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)


@hydra.main(config_path="/home/user/Traj-LLM/imjaegyun/Traj-LLM/configs", config_name="config.yaml")
def main(config: DictConfig):
    train_main(config)


if __name__ == "__main__":
    main()
