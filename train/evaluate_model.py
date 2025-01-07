import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from models.traj_llm_model import TrajLLM
from data.nuscenes_data_loader import NuscenesDataset
import hydra
from omegaconf import DictConfig

def evaluate_model(config: DictConfig):
    # Model initialization
    model = TrajLLM.load_from_checkpoint(config.evaluate.checkpoint_path, config=config)

    # Nuscenes DataLoader setup
    nuscenes_path = config.data.nuscenes_path
    test_dataset = NuscenesDataset(nuscenes_path=nuscenes_path, version="v1.0-mini", split="test")
    test_loader = DataLoader(test_dataset, batch_size=config.evaluate.batch_size, shuffle=False, num_workers=4)

    # Trainer initialization
    trainer = Trainer(gpus=config.train.gpus)

    # Start evaluation
    results = trainer.test(model, test_loader)
    return results

if __name__ == "__main__":
    @hydra.main(config_path="configs", config_name="traj_llm")
    def main(config: DictConfig):
        results = evaluate_model(config)
        print("Evaluation Results:", results)

    main()
