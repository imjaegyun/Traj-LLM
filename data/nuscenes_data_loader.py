# data/nuscenes_data_loader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes

class NuscenesDataset(Dataset):
    def __init__(self, nuscenes_path, version='v1.0-trainval', split='train', transform=None, target_length=15):
        """
        NuscenesDataset 초기화
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

        try:
            # Extract features
            agent_features = self._extract_agent_features(sample)
            lane_features = self._extract_lane_features(sample)

            # Target trajectory points 및 lane labels 생성 (Mock 데이터)
            target_trajectory = self._generate_mock_trajectory()
            lane_label = self._generate_mock_lane_label()

            if self.transform:
                agent_features = self.transform(agent_features)
                lane_features = self.transform(lane_features)

            return {
                'agent_features': torch.tensor(agent_features, dtype=torch.float32),
                'lane_features': torch.tensor(lane_features, dtype=torch.float32),
                'trajectory_points': torch.tensor(target_trajectory, dtype=torch.float32),
                'lane_labels': torch.tensor(lane_label, dtype=torch.long)
            }

        except KeyError as e:
            print(f"[ERROR] Missing key in sample data: {e}. Returning default values.")
            return self._get_default_sample()

    def _extract_agent_features(self, sample):
        """
        샘플에서 agent 특징 추출
        """
        features = np.random.rand(10, 128)  # Mocked: 10 time steps, 128 features
        return self._pad_or_truncate(features, self.target_length)

    def _extract_lane_features(self, sample):
        """
        샘플에서 lane 특징 추출
        """
        features = np.random.rand(15, 128)  # Mocked: 15 lanes, 128 features
        return self._pad_or_truncate(features, self.target_length)

    def _pad_or_truncate(self, features, target_length):
        """
        시퀀스 데이터를 패딩하거나 자릅니다.
        """
        current_length = features.shape[0]
        if current_length < target_length:
            padding = np.zeros((target_length - current_length, features.shape[1]))
            features = np.vstack((features, padding))
        elif current_length > target_length:
            features = features[:target_length]
        return features

    def _generate_mock_trajectory(self):
        """
        Mock 궤적 데이터를 생성합니다.
        """
        return np.random.rand(self.target_length, 2)  # [target_length, 2]

    def _generate_mock_lane_label(self):
        """
        Mock lane label을 생성합니다.
        """
        return np.random.randint(0, 6, size=(self.target_length,))  # [target_length]

    def _get_default_sample(self):
        """
        기본값을 반환합니다.
        """
        return {
            'agent_features': torch.zeros((self.target_length, 128), dtype=torch.float32),
            'lane_features': torch.zeros((self.target_length, 128), dtype=torch.float32),
            'trajectory_points': torch.zeros((self.target_length, 2), dtype=torch.float32),
            'lane_labels': torch.zeros((self.target_length,), dtype=torch.long)
        }

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
            print(f"Batch {i}:")
            print(f"  Agent features shape: {batch['agent_features'].shape}")
            print(f"  Lane features shape: {batch['lane_features'].shape}")
            print(f"  Trajectory points shape: {batch['trajectory_points'].shape}")
            print(f"  Lane labels shape: {batch['lane_labels'].shape}")

            if i == 2:
                break

    except ValueError as e:
        print(f"Dataset Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

