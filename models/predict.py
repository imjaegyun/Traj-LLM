import torch
from pytorch_lightning import Trainer
from train_model import TrajLLM  # 훈련된 모델 클래스
from data_module import NuscenesDataModule  # Nuscenes 데이터 로드 모듈
import argparse
import os
import json

def load_model(checkpoint_path, device):
    """
    훈련된 모델 로드
    """
    model = TrajLLM.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()  # 모델을 평가 모드로 설정
    return model

def run_prediction(model, data_root, batch_size, device, version="v1.0-mini"):
    """
    모델을 사용하여 Nuscenes 데이터셋에서 예측 실행
    """
    # 데이터 로드 모듈 설정
    data_module = NuscenesDataModule(data_root=data_root, batch_size=batch_size, version=version)
    data_module.setup(stage="test")

    # Trainer 초기화 및 예측 실행
    trainer = Trainer(accelerator=device, logger=False, enable_checkpointing=False)
    predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())

    return predictions

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
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"예측 결과가 {output_path}에 저장되었습니다.")

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Trajectory Prediction on Nuscenes")
    parser.add_argument("--checkpoint", type=str, required=True, help="훈련된 모델의 체크포인트 경로")
    parser.add_argument("--data_root", type=str, required=True, help="Nuscenes 데이터셋의 루트 디렉토리")
    parser.add_argument("--output", type=str, default="predictions.json", help="예측 결과를 저장할 파일 경로")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--device", type=str, default="cuda", help="사용할 장치 (cuda 또는 cpu)")
    args = parser.parse_args()

    # 디바이스 설정
    device = args.device if torch.cuda.is_available() else "cpu"

    # 모델 로드
    model = load_model(args.checkpoint, device)

    # 예측 실행
    predictions = run_prediction(model, args.data_root, args.batch_size, device)

    # 예측 결과 저장
    save_predictions(predictions, args.output)

if __name__ == "__main__":
    main()
