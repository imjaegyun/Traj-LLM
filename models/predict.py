import torch
from pytorch_lightning import Trainer
from train.train_model import TrajLLM  # 훈련된 모델 클래스
from data.nuscenes_data_loader import NuscenesDataset  # Nuscenes 데이터 로드 모듈
import argparse
import os
import json
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

def load_model(checkpoint_path, device):
    """
    훈련된 모델 로드
    """
    # 기본 config 경로
    default_config_path = "/home/user/Traj-LLM/imjaegyun/Traj-LLM/configs/config.yaml"
    module_config_path = "/home/user/Traj-LLM/imjaegyun/Traj-LLM/configs/modules/model.yaml"

    # 체크포인트 기준 경로에서 config.yaml 확인
    config_path = os.path.join(os.path.dirname(checkpoint_path), "..", "config.yaml")

    # 기본 config 로드
    config = OmegaConf.load(config_path) if os.path.exists(config_path) else OmegaConf.load(default_config_path)

    # modules 관련 추가 설정 병합
    if os.path.exists(module_config_path):
        module_config = OmegaConf.load(module_config_path)
        config = OmegaConf.merge(config, {"modules": module_config})

    # modules 키가 없거나 필요한 키가 누락된 경우 기본값 추가
    if "modules" not in config:
        config.modules = {}

    if "sparse_encoder" not in config.modules:
        config.modules.sparse_encoder = {"input_dim": 128, "hidden_dim": 128, "output_dim": 128}
    if "output_dim" not in config.modules.sparse_encoder:
        config.modules.sparse_encoder.output_dim = 128  # 기본값 설정

    # 모델 로드
    model = TrajLLM.load_from_checkpoint(checkpoint_path, config=config)
    model.to(device)
    model.eval()  # 모델을 평가 모드로 설정
    return model


def run_prediction(model, data_root, batch_size, device, version="v1.0-trainval"):
    """
    모델을 사용하여 Nuscenes 데이터셋에서 예측 실행 (val 데이터를 test로 사용)
    """
    # 데이터 로드 모듈 설정 (data_root -> nuscenes_path)
    data_module = NuscenesDataset(nuscenes_path=data_root, version=version, split="val")

    # DataLoader 생성
    val_loader = DataLoader(data_module, batch_size=batch_size, shuffle=False, num_workers=4)

    # Trainer 초기화 및 예측 실행
    trainer = Trainer(accelerator=device, logger=False, enable_checkpointing=False)
    predictions = trainer.predict(model, dataloaders=val_loader)

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
    import numpy as np

    def convert_to_serializable(obj):
        """
        Convert non-serializable objects (like ndarray) to serializable types.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    results = []
    for prediction in predictions:
        try:
            results.append({
                "lane_probabilities": convert_to_serializable(prediction.get("lane_probabilities")),
                "lane_predictions": convert_to_serializable(prediction.get("lane_predictions")),
                "mu": convert_to_serializable(prediction.get("mu")),
                "b": convert_to_serializable(prediction.get("b")),
            })
        except Exception as e:
            print(f"Error processing prediction: {e}")

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
