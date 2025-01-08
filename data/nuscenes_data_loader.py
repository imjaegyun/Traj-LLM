import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes

def extract_trajectory(instance, nusc, target_length=15):
    """
    NuScenes 인스턴스에서 궤적 추출
    Args:
        instance: NuScenes의 instance 객체
        nusc: NuScenes 데이터셋 객체
        target_length: 출력 궤적의 목표 길이
    Returns:
        trajectory_points: (target_length, 2) 형태의 궤적 데이터
    """
    trajectory = []
    for ann_token in instance['anns']:
        annotation = nusc.get('sample_annotation', ann_token)
        position = annotation['translation'][:2]  # x, y 좌표만 사용
        trajectory.append(position)

    # 길이 조정 (패딩 또는 자르기)
    trajectory = np.array(trajectory)
    trajectory = pad_or_truncate(trajectory, target_length)
    return trajectory

def extract_lane_label(map_data, position):
    """
    현재 위치에서 가장 가까운 차선을 기반으로 레이블 생성
    Args:
        map_data: NuScenes 맵 객체
        position: 현재 위치 (x, y 좌표)
    Returns:
        lane_label: 가장 가까운 차선 ID 또는 임의의 정수 레이블
    """
    # 위치에서 가장 가까운 차선을 찾는 로직
    closest_lane = None
    closest_distance = float('inf')

    for lane in map_data['lane']:
        lane_position = lane['center']  # 차선의 중심 좌표
        distance = np.linalg.norm(np.array(position) - np.array(lane_position))

        if distance < closest_distance:
            closest_distance = distance
            closest_lane = lane

    # 차선 레이블 반환
    return closest_lane['id'] if closest_lane else 0

def pad_or_truncate(features, target_length):
    """
    시퀀스 길이를 target_length로 맞춤
    Args:
        features: 입력 데이터
        target_length: 목표 시퀀스 길이
    Returns:
        패딩 또는 잘라낸 데이터
    """
    current_length = features.shape[0]
    if current_length < target_length:
        # 패딩 추가
        padding = np.zeros((target_length - current_length, features.shape[1]))
        features = np.vstack((features, padding))
    elif current_length > target_length:
        # 잘라내기
        features = features[:target_length]
    return features

class NuscenesDataset(Dataset):
    def __init__(self, nuscenes_path, version='v1.0-trainval', split='train', transform=None, target_length=15):
        """
        NuscenesDataset 초기화
        Args:
            nuscenes_path: NuScenes 데이터셋의 루트 경로
            version: NuScenes 데이터셋 버전 ('v1.0-trainval')
            split: 데이터 분할 ('train' 또는 'val')
            transform: 데이터 변환 함수
            target_length: 입력 데이터의 시퀀스 길이를 맞출 목표 길이
        """
        self.nusc = NuScenes(version=version, dataroot=nuscenes_path, verbose=True)
        self.split = split
        self.transform = transform
        self.target_length = target_length

        # 모든 scene 로드 및 split 설정
        all_scenes = self.nusc.scene
        print(f"Total scenes in dataset: {len(all_scenes)}")

        train_split = set(range(0, int(0.8 * len(all_scenes))))
        val_split = set(range(int(0.8 * len(all_scenes)), len(all_scenes)))

        if split == 'train':
            self.scenes = [scene for i, scene in enumerate(all_scenes) if i in train_split]
        elif split == 'val':
            self.scenes = [scene for i, scene in enumerate(all_scenes) if i in val_split]
        else:
            raise ValueError(f"Invalid split '{split}'. Choose 'train' or 'val'.")

        print(f"Found {len(self.scenes)} scenes for split '{split}'")

        if len(self.scenes) == 0:
            raise ValueError(f"No scenes found for split '{split}'. Check dataset path and split configuration.")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        first_sample_token = scene['first_sample_token']
        sample = self.nusc.get('sample', first_sample_token)

        # 인스턴스와 맵 데이터 추출
        instance_token = sample['instance_token']
        instance = self.nusc.get('instance', instance_token)
        map_data = self.nusc.get('map', sample['map_token'])

        # 실제 궤적 데이터와 차선 레이블 추출
        target_trajectory = extract_trajectory(instance, self.nusc, self.target_length)
        lane_label = extract_lane_label(map_data, target_trajectory[0])  # 궤적의 첫 위치 기반

        # Mock 데이터 대신 실제 데이터를 사용
        agent_features = self._extract_agent_features(sample)
        lane_features = self._extract_lane_features(sample)

        if self.transform:
            agent_features = self.transform(agent_features)
            lane_features = self.transform(lane_features)

        return {
            'agent_features': torch.tensor(agent_features, dtype=torch.float32),
            'lane_features': torch.tensor(lane_features, dtype=torch.float32),
            'trajectory_points': torch.tensor(target_trajectory, dtype=torch.float32),
            'lane_labels': torch.tensor(lane_label, dtype=torch.long)
        }

    def _extract_agent_features(self, sample):
        """
        샘플에서 agent 특징 추출
        """
        features = np.random.rand(10, 128)  # Mocked: 10 time steps, 128 features
        return pad_or_truncate(features, self.target_length)

    def _extract_lane_features(self, sample):
        """
        샘플에서 lane 특징 추출
        """
        features = np.random.rand(15, 128)  # Mocked: 15 lanes, 128 features
        return pad_or_truncate(features, self.target_length)

# Example usage
if __name__ == "__main__":
    nuscenes_path = "/home/user/data/Nuscenes"
    version = 'v1.0-trainval'
    split = 'train'

    try:
        print("Creating dataset...")
        dataset = NuscenesDataset(nuscenes_path=nuscenes_path, version=version, split=split)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

        print("Iterating through the dataset...")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}: Agent features shape: {batch['agent_features'].shape}")
            print(f"Batch {i}: Lane features shape: {batch['lane_features'].shape}")
            print(f"Batch {i}: Trajectory points shape: {batch['trajectory_points'].shape}")
            print(f"Batch {i}: Lane labels shape: {batch['lane_labels'].shape}")
            if i == 2:
                break

    except ValueError as e:
        print(f"Dataset Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
