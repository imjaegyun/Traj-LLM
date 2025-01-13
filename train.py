# train.py
import hydra
from omegaconf import DictConfig
from train.train_model import train_main
# from train.evaluate_model import main as evaluate_main
# from train.hyperparam_tuning import main as tuning_main

@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.1")
def main(config: DictConfig):
    # Select task based on config
    task = config.get("task", "train")  # Default to 'train' if not specified

    if task == "train":
        print("Starting training...")
        train_main(config)  # Pass DictConfig to train_main
    elif task == "evaluate":
        print("Starting evaluation...")
        # evaluate_main(config)
        print("Evaluation functionality is not yet implemented.")
    elif task == "tune":
        print("Starting hyperparameter tuning...")
        # tuning_main(config)
        print("Hyperparameter tuning functionality is not yet implemented.")
    else:
        raise ValueError(f"Unknown task: {task}. Choose from ['train', 'evaluate', 'tune'].")

if __name__ == "__main__":
    main()
