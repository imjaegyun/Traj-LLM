# data/nuscenes_data_loader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import logging

logger = logging.getLogger(__name__)

class NuscenesDataset(Dataset):
    def __init__(self, nuscenes_path, version='v1.0-trainval', split='train', target_length=240):
        """
        Args:
            nuscenes_path (str): Path to the NuScenes dataset directory
            version (str): NuScenes version (e.g., 'v1.0-trainval')
            split (str): 'train' or 'val'
            target_length (int): Number of future frames to predict
        """
        self.output_length = 6
        self.target_length = target_length
        self.nusc = NuScenes(version=version, dataroot=nuscenes_path, verbose=True)
        self.nusc_map = NuScenesMap(dataroot=nuscenes_path, map_name="singapore-onenorth")
        self.split = split

        all_scenes = self.nusc.scene
        train_split = set(range(0, int(0.8 * len(all_scenes))))
        val_split = set(range(int(0.8 * len(all_scenes)), len(all_scenes)))

        if split == 'train':
            self.scenes = [scene for i, scene in enumerate(all_scenes) if i in train_split]
        elif split == 'val':
            self.scenes = [scene for i, scene in enumerate(all_scenes) if i in val_split]
        else:
            raise ValueError(f"Invalid split '{split}'. Choose 'train' or 'val'.")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        first_sample_token = scene['first_sample_token']
        sample = self.nusc.get('sample', first_sample_token)

        try:
            agent_state = self._extract_agent_state(sample)
            lane_features = self._extract_lane_features(sample)
            trajectory_points = self._extract_trajectory_points(sample)
            lane_labels = self._extract_lane_labels(sample)  # lane_labels 추가

            # lane_features가 모두 0인 경우 배치 제외
            if np.all(lane_features == 0):
                raise ValueError("Invalid lane features. Skipping sample.")

            return {
                'agent_features': torch.tensor(agent_state, dtype=torch.float32),
                'lane_features': torch.tensor(lane_features, dtype=torch.float32),
                'trajectory_points': torch.tensor(trajectory_points, dtype=torch.float32),
                'lane_labels': torch.tensor(lane_labels, dtype=torch.long),  # lane_labels 포함
            }
        except Exception as e:
            print(f"[ERROR] Failed to process sample {idx}: {e}. Skipping.")
            return self._get_default_sample()

    def _extract_lane_features(self, sample):
        try:
            sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])

            ego_x, ego_y = ego_pose['translation'][0], ego_pose['translation'][1]

            # NuScenesMap API를 사용하여 반경 50m 내 차선 검색
            lanes_in_radius = self.nusc_map.get_records_in_radius(
                ego_x, ego_y, 50.0, ['lane']
            )

            if not lanes_in_radius['lane']:
                return np.zeros((self.target_length, 128))

            lane_features = []
            for lane_token in lanes_in_radius['lane']:
                lane_record = self.nusc_map.get('lane', lane_token)
                width = lane_record.get('width', 0.0)
                length = lane_record.get('length', 0.0)
                lane_features.append([width, length])

            lane_features = np.array(lane_features)

            # (N, 2) -> 128차원으로 패딩. (실제론 더욱 복잡한 차원 추출 가능)
            lane_features = np.pad(
                lane_features,
                ((0, max(0, self.target_length - lane_features.shape[0])), (0, 126)),
                mode='constant'
            )
            lane_features = lane_features[:self.target_length, :128]

            return lane_features

        except Exception as e:
            print(f"[ERROR] Failed to extract lane features: {e}. Using zeros.")
            return np.zeros((self.target_length, 128))

    def _extract_lane_labels(self, sample):
        """
        Extract lane labels for the current sample. Implement the logic based on your task.
        This is a placeholder function.
        """
        # Placeholder implementation: 예시로 모든 lane_labels를 0으로 설정
        # 실제로는 차선의 카테고리나 기타 정보를 기반으로 레이블을 설정해야 합니다.
        lane_labels = np.zeros((self.target_length,), dtype=int)
        return lane_labels

    def _extract_agent_state(self, sample):
        try:
            trajectory = []
            velocity = []

            current_sample = sample
            for _ in range(self.output_length):
                sd_rec = self.nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
                ego_pose = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])

                x, y = ego_pose['translation'][:2]
                trajectory.append([x, y])

                if current_sample['next'] != "":
                    next_sample = self.nusc.get('sample', current_sample['next'])
                    next_sd_rec = self.nusc.get('sample_data', next_sample['data']['LIDAR_TOP'])
                    next_pose = self.nusc.get('ego_pose', next_sd_rec['ego_pose_token'])

                    dt = (next_pose['timestamp'] - ego_pose['timestamp']) / 1e6
                    nx, ny = next_pose['translation'][:2]
                    vx = (nx - x) / dt
                    vy = (ny - y) / dt
                else:
                    vx, vy = 0.0, 0.0

                velocity.append([vx, vy])

                if current_sample['next'] != "":
                    current_sample = self.nusc.get('sample', current_sample['next'])
                else:
                    break

            trajectory = np.array(trajectory)
            velocity = np.array(velocity)

            if len(trajectory) < self.output_length:
                last_traj = trajectory[-1].copy()
                last_vel = velocity[-1].copy()
                diff = self.output_length - len(trajectory)
                trajectory = np.concatenate([trajectory, np.tile(last_traj, (diff, 1))], axis=0)
                velocity = np.concatenate([velocity, np.tile(last_vel, (diff, 1))], axis=0)

            agent_state = np.hstack((trajectory, velocity))  # (6, 4)
            return agent_state

        except Exception as e:
            print(f"[ERROR] Failed to extract agent state: {e}. Using zeros.")
            return np.zeros((self.output_length, 4))

    def _extract_trajectory_points(self, sample):
        """
        실제 미래 궤적 (ego) 위치를 self.target_length 프레임만큼 추출.
        """
        points = []
        current_sample = sample
        for _ in range(self.target_length):
            sd_rec = self.nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
            ego_pose = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])

            x, y = ego_pose['translation'][:2]
            points.append([x, y])

            if current_sample['next'] == "":
                break
            else:
                current_sample = self.nusc.get('sample', current_sample['next'])

        # 부족하면 마지막 위치로 패딩
        if len(points) < self.target_length:
            last_xy = points[-1]
            diff = self.target_length - len(points)
            for _ in range(diff):
                points.append(last_xy)

        return np.array(points)

    def _get_default_sample(self):
        return {
            'agent_features': torch.zeros(self.output_length, 4, dtype=torch.float32),
            'lane_features': torch.zeros(self.target_length, 128, dtype=torch.float32),
            'trajectory_points': torch.zeros(self.target_length, 2, dtype=torch.float32),
            'lane_labels': torch.zeros(self.target_length, dtype=torch.long),  # lane_labels 포함
        }

class NuscenesDatasetFiltered(NuscenesDataset):
    def __init__(self, nuscenes_path, version='v1.0-trainval', split='train', target_length=240):
        super().__init__(nuscenes_path=nuscenes_path, version=version, split=split, target_length=target_length)
        original_length = len(self.scenes)
        self.scenes = [scene for scene in self.scenes if self.is_valid(scene)]
        filtered_length = len(self.scenes)
        logger.info(f"Filtered dataset from {original_length} to {filtered_length} scenes.")

    def is_valid(self, scene):
        try:
            first_sample_token = scene['first_sample_token']
            sample = self.nusc.get('sample', first_sample_token)
            lane_features = self._extract_lane_features(sample)
            if np.all(lane_features == 0):
                logger.error(f"Scene {scene['name']} has invalid lane features.")
                return False
            return True
        except Exception as e:
            logger.error(f"Scene {scene['name']} failed validation: {e}")
            return False
