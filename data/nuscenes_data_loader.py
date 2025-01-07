import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

class NuscenesDataset(Dataset):
    def __init__(self, nuscenes_path, version='v1.0-mini', split='train', transform=None):
        self.nusc = NuScenes(version=version, dataroot=nuscenes_path, verbose=True)
        self.split = split
        self.transform = transform

        # Filter scenes by split
        self.scenes = [scene for scene in self.nusc.scene if split in scene['name']]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        first_sample_token = scene['first_sample_token']

        # Extract sample data
        sample = self.nusc.get('sample', first_sample_token)

        # Agent and Lane features (simplified example)
        agent_features = self._extract_agent_features(sample)
        lane_features = self._extract_lane_features(sample)

        # Target trajectory points and lane labels (mocked for simplicity)
        target_trajectory = np.random.rand(1, 2)  # Mocked target x, y
        lane_label = np.random.randint(0, 5)  # Mocked lane ID

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
        # Placeholder: Extract agent features from sample data
        return np.random.rand(10, 128)  # Mocked 10 time steps, 128 features

    def _extract_lane_features(self, sample):
        # Placeholder: Extract lane features from map data
        return np.random.rand(15, 128)  # Mocked 15 lanes, 128 features

# Example usage
if __name__ == "__main__":
    nuscenes_path = "path/to/nuscenes/data"
    dataset = NuscenesDataset(nuscenes_path=nuscenes_path, version='v1.0-mini', split='train')

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    for batch in dataloader:
        print("Agent features shape:", batch['agent_features'].shape)
        print("Lane features shape:", batch['lane_features'].shape)
        print("Trajectory points shape:", batch['trajectory_points'].shape)
        print("Lane labels shape:", batch['lane_labels'].shape)
        break
