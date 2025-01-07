import hydra
from omegaconf import DictConfig
from train.train_model import main as train_model
from train.evaluate_model import main as evaluate_model
from train.hyperparam_tuning import main as hyperparam_tuning

@hydra.main(config_path="configs", config_name="traj_llm")
def main(config: DictConfig):
    # Select task based on config
    task = config.task

    if task == "train":
        print("Starting training...")
        train_model(config)
    elif task == "evaluate":
        print("Starting evaluation...")
        evaluate_model(config)
    elif task == "tune":
        print("Starting hyperparameter tuning...")
        hyperparam_tuning(config)
    else:
        raise ValueError(f"Unknown task: {task}. Choose from ['train', 'evaluate', 'tune'].")

if __name__ == "__main__":
    main()
