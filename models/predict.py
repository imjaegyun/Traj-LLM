import torch
from pytorch_lightning import Trainer
from train.train_model import TrajLLM  # 훈련된 모델 클래스
from data.nuscenes_data_loader import NuscenesDataset  # Nuscenes 데이터 로드 모듈
import argparse
import os
import json
import numpy as np
from omegaconf import OmegaConf

def load_model(checkpoint_path, device):
    """
    훈련된 모델 로드
    """
    # config.yaml 경로를 기반으로 설정 로드
    default_config_path = "/home/user/Traj-LLM/imjaegyun/Traj-LLM/configs/config.yaml"
    config_path = os.path.join(os.path.dirname(checkpoint_path), "..", "config.yaml")
    config = OmegaConf.load(config_path) if os.path.exists(config_path) else OmegaConf.load(default_config_path)

    # modules 키가 없을 경우 기본값 추가
    if 'modules' not in config:
        config.modules = {
            "sparse_encoder": {"input_dim": 128, "hidden_dim": 256},
            "high_level_interaction_model": {"hidden_dim": 256},
            "lane_aware_probability_learning": {"output_dim": 6},
            "multimodal_laplace_decoder": {"hidden_dim": 128}
        }

    model = TrajLLM.load_from_checkpoint(checkpoint_path, config=config)
    model.to(device)
    model.eval()  # 모델을 평가 모드로 설정
    return model

def run_prediction(model, data_root, batch_size, device, version="v1.0-mini"):
    """
    모델을 사용하여 Nuscenes 데이터셋에서 예측 실행 (val 데이터를 test로 사용)
    """
    # 데이터 로드 모듈 설정
    data_module = NuscenesDataset(data_root=data_root, batch_size=batch_size, version=version)
    data_module.setup(stage="val")  # val 데이터를 test 대용으로 사용

    # Trainer 초기화 및 예측 실행
    trainer = Trainer(accelerator=device, logger=False, enable_checkpointing=False)
    predictions = trainer.predict(model, dataloaders=data_module.val_dataloader())  # val_dataloader() 호출

    return predictions

def compute_metrics(predictions, ground_truths, threshold=2.0):
    """
    ADE, FDE, MR 평가 지표 계산
    """
    ade = 0.0
    fde = 0.0
    misses = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        pred_traj = np.array(pred["trajectory"])
        gt_traj = np.array(gt["trajectory"])

        # ADE 계산
        ade += np.mean(np.linalg.norm(pred_traj - gt_traj, axis=1))

        # FDE 계산
        fde += np.linalg.norm(pred_traj[-1] - gt_traj[-1])

        # Miss Rate 계산
        miss = np.linalg.norm(pred_traj[-1] - gt_traj[-1]) > threshold
        misses += miss

    ade /= total
    fde /= total
    mr = misses / total

    return {"ADE": ade, "FDE": fde, "MR": mr}

def save_predictions(predictions, output_path):
    """
    예측 결과 저장
    """
    results = []
    for prediction in predictions:
        results.append({
            "lane_probabilities": prediction["lane_probabilities"].tolist(),
            "lane_predictions": prediction["lane_predictions"].tolist(),
            "mu": prediction["mu"].tolist(),
            "b": prediction["b"].tolist(),
            "trajectory": prediction["trajectory"].tolist(),
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"예측 결과가 {output_path}에 저장되었습니다.")

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Trajectory Prediction on Nuscenes")
    parser.add_argument("--checkpoint", type=str, required=True, help="훈련된 모델의 체크포인트 경로")
    parser.add_argument("--output", type=str, default="predictions.json", help="예측 결과를 저장할 파일 경로")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--device", type=str, default="cuda", help="사용할 장치 (cuda 또는 cpu)")
    args = parser.parse_args()

    # 고정된 Nuscenes 데이터 경로
    data_root = "/home/user/data/Nuscenes"

    # 디바이스 설정
    device = args.device if torch.cuda.is_available() else "cpu"

    # 모델 로드
    model = load_model(args.checkpoint, device)

    # 예측 실행
    predictions = run_prediction(model, data_root, args.batch_size, device)

    # Ground Truth 데이터 로드
    ground_truths = [{"trajectory": p["ground_truth"]} for p in predictions if "ground_truth" in p]

    if ground_truths:
        # 평가 지표 계산
        metrics = compute_metrics(predictions, ground_truths)

        # 결과 출력
        print("Evaluation Metrics:")
        print(f"ADE: {metrics['ADE']:.4f}")
        print(f"FDE: {metrics['FDE']:.4f}")
        print(f"MR: {metrics['MR']:.4f}")

    # 예측 결과 저장
    save_predictions(predictions, args.output)

if __name__ == "__main__":
    main()
