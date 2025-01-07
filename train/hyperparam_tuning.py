import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models.traj_llm_model import TrajLLM
from data.nuscenes_data_loader import NuscenesDataset
import hydra
from omegaconf import DictConfig

def objective(trial, config):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512, step=64)

    # Update config with suggested values
    config.train.lr = learning_rate
    config.train.batch_size = batch_size
    config.sparse_encoder.hidden_dim = hidden_dim

    # DataLoader setup
    nuscenes_path = config.data.nuscenes_path
    train_dataset = NuscenesDataset(nuscenes_path=nuscenes_path, version="v1.0-mini", split="train")
    val_dataset = NuscenesDataset(nuscenes_path=nuscenes_path, version="v1.0-mini", split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = TrajLLM(config)

    # WandB Logger
    wandb_logger = WandbLogger(project="Traj-LLM-Tuning")

    # Trainer setup
    trainer = Trainer(
        max_epochs=config.train.epochs,
        gpus=config.train.gpus,
        logger=wandb_logger
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Validation loss as the objective
    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss

@hydra.main(config_path="configs", config_name="traj_llm")
def main(config: DictConfig):
    # Create Optuna study
    study = optuna.create_study(direction="minimize")

    # Optimize
    study.optimize(lambda trial: objective(trial, config), n_trials=20)

    # Log best parameters
    print("Best trial:", study.best_trial.params)

if __name__ == "__main__":
    main()