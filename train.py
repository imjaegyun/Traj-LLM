import hydra
from omegaconf import DictConfig
from train.train_model import  train_main
from train.evaluate_model import main as evaluate_main
from train.hyperparam_tuning import main as tuning_main

@hydra.main(config_path="/home/user/Traj-LLM/imjaegyun/Traj-LLM/configs", config_name="config.yaml")
def main(config: DictConfig):
    # Select task based on config
    task = config.task

    if task == "train":
        print("Starting training...")
        train_main(config)  # train_main에 DictConfig 전달
    elif task == "evaluate":
        print("Starting evaluation...")
        evaluate_main(config)
    elif task == "tune":
        print("Starting hyperparameter tuning...")
        tuning_main(config)
    else:
        raise ValueError(f"Unknown task: {task}. Choose from ['train', 'evaluate', 'tune'].")

if __name__ == "__main__":
    main()
