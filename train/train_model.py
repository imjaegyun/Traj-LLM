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
from data.nuscenes_data_loader import NuscenesDatasetFiltered
import hydra
from omegaconf import DictConfig, OmegaConf
from models.mamba import MambaLayer
from torch.nn.utils.rnn import pad_sequence
import logging

# 설정 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def custom_collate(batch):
    # 유효하지 않은 샘플(기본값)을 배제
    batch = [item for item in batch if not torch.equal(item['agent_features'], torch.zeros_like(item['agent_features']))]
    
    if not batch:
        # 모든 샘플이 유효하지 않은 경우 기본값 반환
        return {
            'agent_features': torch.zeros(0, 6, 4, dtype=torch.float32),
            'lane_features': torch.zeros(0, 240, 128, dtype=torch.float32),
            'trajectory_points': torch.zeros(0, 240, 2, dtype=torch.float32),
            'lane_labels': torch.zeros(0, 240, dtype=torch.long),
        }
    
    agent_features = [item['agent_features'] for item in batch]
    lane_features = [item['lane_features'] for item in batch]
    traj_points = [item['trajectory_points'] for item in batch]
    lane_labels = [item['lane_labels'] for item in batch]
    
    # 시퀀스 길이 패딩 (필요 시)
    traj_points_padded = pad_sequence(traj_points, batch_first=True, padding_value=0)
    lane_labels_padded = pad_sequence(lane_labels, batch_first=True, padding_value=0)
    
    return {
        'agent_features': torch.stack(agent_features),
        'lane_features': torch.stack(lane_features),
        'trajectory_points': traj_points_padded,
        'lane_labels': lane_labels_padded,
    }

class TrajLLM(pl.LightningModule):
    def __init__(self, config):
        super(TrajLLM, self).__init__()

        self.config = config
        self.k_values = config.train.k_values
        self.learning_rate = config.train.lr

        self.sparse_encoder = SparseContextEncoder(
            input_dim=config.modules.sparse_encoder.input_dim,
            hidden_dim=config.modules.sparse_encoder.hidden_dim,
            output_dim=config.modules.sparse_encoder.output_dim
        )

        if not hasattr(config.modules.high_level_model, "llm_model_name"):
            raise ValueError("llm_model_name is missing in config.modules.high_level_model")

        self.high_level_model = HighLevelInteractionModel(
            llm_model_name=config.modules.high_level_model.llm_model_name,
            input_dim=config.modules.high_level_model.input_dim,
            output_dim=config.modules.high_level_model.output_dim
        )

        self.lane_probability_model = LaneAwareProbabilityLearning(
            agent_dim=config.modules.lane_probability.agent_dim,
            lane_dim=config.modules.lane_probability.lane_dim,
            hidden_dim=config.modules.lane_probability.hidden_dim,
            num_lanes=config.modules.lane_probability.num_lanes
        )

        self.laplace_decoder = MultimodalLaplaceDecoder(
            input_dim=config.modules.laplace_decoder.input_dim,
            output_dim=config.modules.laplace_decoder.output_dim
        )

        self.mamba_layer = MambaLayer(
            input_dim=config.modules.mamba.input_dim,
            hidden_dim=config.modules.mamba.hidden_dim,
            num_blocks=config.modules.mamba.num_blocks
        ) if hasattr(config.modules, "mamba") else None

        self.reset_parameters()
        torch.set_float32_matmul_precision("medium")
        self.validation_outputs = []

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, agent_inputs, lane_inputs):
        device = agent_inputs.device
        sparse_features = self.sparse_encoder(agent_inputs, lane_inputs)
        high_level_features = self.high_level_model(sparse_features, device=device)
        lane_probabilities, lane_predictions = self.lane_probability_model(high_level_features, lane_inputs)
        if self.mamba_layer:
            high_level_features = self.mamba_layer(high_level_features)
        pi, mu, b, uncertainty = self.laplace_decoder(high_level_features, lane_probabilities)
        return lane_probabilities, lane_predictions, pi, mu, b, uncertainty

    def compute_laplace_loss(self, pi, mu, b, traj_points):
        batch_size, seq_len, num_modes, _ = mu.size()
        gt_expanded = traj_points.unsqueeze(2)
        l2_distances = torch.norm(mu - gt_expanded, dim=-1)
        k_star = torch.argmin(l2_distances, dim=-1, keepdim=True)
        k_star_mask = torch.zeros_like(l2_distances).scatter_(-1, k_star, 1.0)
        b = torch.clamp(b, min=1e-6)
        laplace_density = -torch.abs(mu - gt_expanded) / b - torch.log(2 * b)
        laplace_density = torch.sum(laplace_density, dim=-1)
        weighted_density = pi + laplace_density
        selected_density = torch.sum(weighted_density * k_star_mask, dim=-1)
        reg_loss = -torch.mean(selected_density)
        reg_loss = torch.clamp(reg_loss, min=0)
        return reg_loss

    def training_step(self, batch, batch_idx):
        agent_inputs = batch['agent_features']
        lane_inputs = batch['lane_features']
        traj_points = batch.get('trajectory_points', None)
        lane_labels = batch.get('lane_labels', None)

        lane_probs, lane_preds, pi, mu, b, uncertainty = self(agent_inputs, lane_inputs)

        reg_loss = torch.tensor(0.0, device=self.device)
        if traj_points is not None and mu is not None and b is not None:
            reg_loss = self.compute_laplace_loss(pi, mu, b, traj_points[:, :6, :])

        cls_loss = torch.tensor(0.0, device=self.device)
        if pi is not None:
            cls_loss_fn = nn.CrossEntropyLoss()
            cls_loss = cls_loss_fn(pi.view(-1, pi.size(-1)), torch.argmax(pi, dim=-1).view(-1))

        lane_loss = torch.tensor(0.0, device=self.device)
        if lane_labels is not None:
            lane_loss_fn = nn.CrossEntropyLoss()
            lane_loss = lane_loss_fn(
                lane_probs.view(-1, lane_probs.size(-1)),
                lane_labels.view(-1)
            )

        lambda_lane = self.config.train.lambda_lane
        total_loss = lambda_lane * lane_loss + reg_loss + cls_loss

        logger.info(f"Training Step {batch_idx}: Total Loss = {total_loss.item()}, Lane Loss = {lane_loss.item()}, Reg Loss = {reg_loss.item()}, Cls Loss = {cls_loss.item()}")

        self.log("train/lane_loss", lane_loss, on_step=True, on_epoch=True)
        self.log("train/reg_loss", reg_loss, on_step=True, on_epoch=True)
        self.log("train/cls_loss", cls_loss, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        agent_inputs = batch['agent_features']
        lane_inputs = batch['lane_features']
        traj_points = batch.get('trajectory_points', None)
        lane_labels = batch.get('lane_labels', None)

        lane_probs, lane_preds, pi, mu, b, uncertainty = self(agent_inputs, lane_inputs)

        reg_loss = torch.tensor(0.0, device=self.device)
        if traj_points is not None and mu is not None and b is not None:
            reg_loss = self.compute_laplace_loss(pi, mu, b, traj_points[:, :6, :])

        cls_loss = torch.tensor(0.0, device=self.device)
        if pi is not None:
            cls_loss_fn = nn.CrossEntropyLoss()
            cls_loss = cls_loss_fn(pi.view(-1, pi.size(-1)), torch.argmax(pi, dim=-1).view(-1))

        lane_loss = torch.tensor(0.0, device=self.device)
        if lane_labels is not None:
            lane_loss_fn = nn.CrossEntropyLoss()
            lane_loss = lane_loss_fn(
                lane_probs.view(-1, lane_probs.size(-1)),
                lane_labels.view(-1)
            )

        lambda_lane = self.config.train.lambda_lane
        total_loss = lambda_lane * lane_loss + reg_loss + cls_loss

        logger.info(f"Validation Step {batch_idx}: Total Loss = {total_loss.item()}, Lane Loss = {lane_loss.item()}, Reg Loss = {reg_loss.item()}, Cls Loss = {cls_loss.item()}")

        self.validation_outputs.append({
            "reg_loss": reg_loss,
            "cls_loss": cls_loss,
            "lane_loss": lane_loss,
            "total_loss": total_loss,
            "pi": pi,
            "mu": mu,
            "b": b,
            "traj_points": traj_points
        })

        self.log("val/lane_loss", lane_loss, on_epoch=True)
        self.log("val/reg_loss", reg_loss, on_epoch=True)
        self.log("val/cls_loss", cls_loss, on_epoch=True)
        self.log("val/total_loss", total_loss, on_epoch=True)

        return total_loss

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            logger.warning("No validation outputs to process.")
            return

        all_reg_loss = torch.stack([x['reg_loss'] for x in self.validation_outputs]).mean()
        all_cls_loss = torch.stack([x['cls_loss'] for x in self.validation_outputs]).mean()
        all_lane_loss = torch.stack([x['lane_loss'] for x in self.validation_outputs]).mean()
        all_total_loss = torch.stack([x['total_loss'] for x in self.validation_outputs]).mean()

        all_pi = torch.cat([x['pi'] for x in self.validation_outputs], dim=0)
        all_mu = torch.cat([x['mu'] for x in self.validation_outputs], dim=0)
        all_b = torch.cat([x['b'] for x in self.validation_outputs], dim=0)
        all_traj_points = torch.cat([x['traj_points'] for x in self.validation_outputs], dim=0)

        metrics = self._compute_metrics(all_mu, all_traj_points)
        logger.info(f"Validation Metrics: minADE = {metrics['minADE']}, minFDE = {metrics['minFDE']}, MR = {metrics['MR']}")

        self.log("val/minADE", metrics['minADE'], on_epoch=True)
        self.log("val/minFDE", metrics['minFDE'], on_epoch=True)
        self.log("val/MR", metrics['MR'], on_epoch=True)

        self.validation_outputs.clear()

    def _compute_metrics(self, predicted_trajs, gt_trajs):
        # 시퀀스 길이 맞추기
        seq_len_pred = predicted_trajs.size(1)
        seq_len_gt = gt_trajs.size(1)

        if seq_len_pred > seq_len_gt:
            predicted_trajs = predicted_trajs[:, :seq_len_gt, :]
            logger.warning(f"Predicted trajectories truncated from {seq_len_pred} to {seq_len_gt}.")
        elif seq_len_pred < seq_len_gt:
            gt_trajs = gt_trajs[:, :seq_len_pred, :]
            logger.warning(f"Ground truth trajectories truncated from {seq_len_gt} to {seq_len_pred}.")

        # 텐서 크기 로깅
        logger.info(f"predicted_trajs shape: {predicted_trajs.shape}")
        logger.info(f"gt_trajs shape: {gt_trajs.shape}")

        # 크기 일치 확인
        assert predicted_trajs.size() == gt_trajs.size(), f"Mismatch: {predicted_trajs.size()} vs {gt_trajs.size()}"

        # minADE 계산
        minADE = torch.mean(torch.min(torch.norm(predicted_trajs - gt_trajs.unsqueeze(2), dim=-1), dim=-1)[0]).item()

        # minFDE 계산
        minFDE = torch.mean(torch.norm(predicted_trajs[:, -1, :] - gt_trajs[:, -1, :], dim=-1)).item()

        # MR 계산
        MR = torch.mean((torch.norm(predicted_trajs[:, -1, :] - gt_trajs[:, -1, :], dim=-1) > 2.0).float()).item()

        return {"minADE": minADE, "minFDE": minFDE, "MR": MR}

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
    train_dataset = NuscenesDatasetFiltered(nuscenes_path=nuscenes_path, version="v1.0-trainval", split="train")
    val_dataset = NuscenesDatasetFiltered(nuscenes_path=nuscenes_path, version="v1.0-trainval", split="val")

    batch_size = config.train.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate
    )

    trainer = pl.Trainer(
        max_epochs=config.train.epochs,
        accelerator='gpu',
        devices=config.train.gpus,
        logger=wandb_logger,
        gradient_clip_val=config.train.gradient_clip_val,
        callbacks=[]  # 필요 시 콜백 추가
    )

    trainer.fit(model, train_loader, val_loader)

@hydra.main(version_base="1.1",
            config_path="/home/user/Traj-LLM/imjaegyun/Traj-LLM/configs",
            config_name="config.yaml")
def main(config: DictConfig):
    logger.info("Starting training...")
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    train_main(config)

if __name__ == "__main__":
    main()
