# data/Nuscenes_data_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.geometry import LineString, Point
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NuscenesDatasetFiltered(Dataset):
    def __init__(self, nuscenes_path, version='v1.0-trainval',
                 split='train', target_length=12, num_agents=6,
                 num_lanes=6, radius_m=50.0):
        super().__init__()
        self.target_length = target_length
        self.num_agents = num_agents
        self.num_lanes = num_lanes
        self.radius_m = radius_m

        self.nusc = NuScenes(version=version, dataroot=nuscenes_path, verbose=True)
        # Assuming map name is 'singapore-onenorth'
        self.nusc_map = NuScenesMap(dataroot=nuscenes_path, map_name='singapore-onenorth')

        self.split = split
        all_scenes = self.nusc.scene
        total_scenes = len(all_scenes)
        train_split = set(range(0, int(0.8 * total_scenes)))
        val_split   = set(range(int(0.8 * total_scenes), total_scenes))

        if split == 'train':
            self.scenes = [s for i,s in enumerate(all_scenes) if i in train_split]
        elif split == 'val':
            self.scenes = [s for i,s in enumerate(all_scenes) if i in val_split]
        else:
            raise ValueError(f"Invalid split={split}")

        logger.info(f"[NuscenesDatasetFiltered] Creating dataset for '{split}' with {len(self.scenes)} scenes.")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        first_sample_token = scene['first_sample_token']
        sample = self.nusc.get('sample', first_sample_token)

        try:
            agent_features = self._extract_agent_state(sample)     # [num_agents,4]
            lane_features  = self._extract_lane_features(sample)   # [num_lanes,4]
            trajectory_pts = self._extract_trajectory_points(sample)# [target_length,2]
            lane_labels    = self._extract_lane_labels(sample)      # [num_lanes]

            ### DEBUG ###
            # print(f"[DEBUG] sample={idx}, agent_features={agent_features.shape}, lane_features={lane_features.shape}, traj={trajectory_pts.shape}, lane_labels={lane_labels.shape}")

            return {
                'agent_features': torch.tensor(agent_features, dtype=torch.float32),
                'lane_features':  torch.tensor(lane_features,  dtype=torch.float32),
                'trajectory_points': torch.tensor(trajectory_pts, dtype=torch.float32),
                'lane_labels': torch.tensor(lane_labels, dtype=torch.long)  # Added lane_labels
            }
        except Exception as e:
            logger.error(f"[NuscenesDatasetFiltered] idx={idx}, error={e}. Return default.")
            return self._default_item()

    def _extract_agent_state(self, sample):
        """
        agent_features: (num_agents=6, 4) = [x,y,vx,vy]
        """
        out_len = self.num_agents
        feats = []
        curr_sample = sample
        for _ in range(out_len):
            sd = self.nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
            pose = self.nusc.get('ego_pose', sd['ego_pose_token'])
            x, y = pose['translation'][:2]

            vx, vy = 0.0, 0.0
            if curr_sample['next']:
                nxt = self.nusc.get('sample', curr_sample['next'])
                sd2 = self.nusc.get('sample_data', nxt['data']['LIDAR_TOP'])
                pose2 = self.nusc.get('ego_pose', sd2['ego_pose_token'])
                dt = (pose2['timestamp'] - pose['timestamp'])/1e6
                if dt > 0:
                    nx, ny = pose2['translation'][:2]
                    vx = (nx - x)/dt
                    vy = (ny - y)/dt

            feats.append([x, y, vx, vy])
            if curr_sample['next']:
                curr_sample = self.nusc.get('sample', curr_sample['next'])
            else:
                break

        feats = np.array(feats)
        if feats.shape[0] < out_len:
            last = feats[-1].copy()
            diff = out_len - feats.shape[0]
            feats = np.concatenate([feats, np.tile(last, (diff, 1))], axis=0)
        return feats

    def _extract_lane_features(self, sample):
        """
        lane_features: (num_lanes=6, 4) = [nx, ny, dx, dy]
        """
        sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose = self.nusc.get('ego_pose', sd['ego_pose_token'])
        ego_x, ego_y = pose['translation'][:2]

        recs = self.nusc_map.get_records_in_radius(ego_x, ego_y, self.radius_m, ['lane','lane_connector'])
        lane_tokens = recs.get('lane',[]) + recs.get('lane_connector',[])
        if len(lane_tokens) == 0:
            return np.zeros((self.num_lanes,4), dtype=np.float32)

        lane_candidates = []
        for lane_token in lane_tokens:
            try:
                centerline_pts = self.nusc_map.discretize_lane(lane_token, resolution_meters=1.0)
                line_geom = LineString(centerline_pts)
                p_ego = Point(ego_x, ego_y)
                dist = line_geom.distance(p_ego)
                nearest_pt = line_geom.interpolate(line_geom.project(p_ego))
                nx, ny = nearest_pt.x, nearest_pt.y

                s = line_geom.project(p_ego)
                ds = 0.1
                s_next = min(s + ds, line_geom.length)
                next_pt = line_geom.interpolate(s_next)
                dx, dy = (next_pt.x - nx), (next_pt.y - ny)
                norm = max(1e-9, (dx**2 + dy**2)**0.5)
                dx, dy = dx / norm, dy / norm

                lane_candidates.append((dist, nx, ny, dx, dy))
            except:
                continue

        lane_candidates.sort(key=lambda x: x[0])
        feats = []
        for i in range(self.num_lanes):
            if i < len(lane_candidates):
                _, nx, ny, dx, dy = lane_candidates[i]
                feats.append([nx, ny, dx, dy])
            else:
                feats.append([0, 0, 0, 0])

        return np.array(feats, dtype=np.float32)

    def _extract_trajectory_points(self, sample):
        """
        traj_points: [target_length=12, 2] = (x, y)
        """
        pts = []
        cur = sample
        for _ in range(self.target_length):
            sd = self.nusc.get('sample_data', cur['data']['LIDAR_TOP'])
            pose = self.nusc.get('ego_pose', sd['ego_pose_token'])
            x, y = pose['translation'][:2]
            pts.append([x, y])
            if not cur['next']:
                break
            cur = self.nusc.get('sample', cur['next'])

        pts = np.array(pts)
        if pts.shape[0] < self.target_length:
            last_xy = pts[-1].copy()
            diff = self.target_length - pts.shape[0]
            pts = np.concatenate([pts, np.tile(last_xy, (diff, 1))], axis=0)
        return pts

    def _extract_lane_labels(self, sample):
        """
        Extract lane labels from the sample.
        The implementation depends on how lane labels are stored in your dataset.
        """
        # Example placeholder:
        # Assume each sample has a list of lane indices or identifiers
        # Modify this method based on your actual data structure
        lane_tokens = sample.get('lane_tokens', None)
        if lane_tokens is None:
            # Assign default lane labels if not available
            lane_labels = [0] * self.num_lanes
        else:
            # Convert lane_tokens to lane indices or class labels as needed
            # Here, we assign unique indices for demonstration
            lane_labels = [int(token[-4:], 16) % self.num_lanes for token in lane_tokens[:self.num_lanes]]
            # Ensure lane_labels has exactly num_lanes elements
            if len(lane_labels) < self.num_lanes:
                lane_labels += [0] * (self.num_lanes - len(lane_labels))
            else:
                lane_labels = lane_labels[:self.num_lanes]
        return lane_labels

    def _default_item(self):
        return {
            'agent_features': torch.zeros((self.num_agents,4), dtype=torch.float32),
            'lane_features':  torch.zeros((self.num_lanes,4), dtype=torch.float32),
            'trajectory_points': torch.zeros((self.target_length,2), dtype=torch.float32),
            'lane_labels': torch.zeros((self.num_lanes,), dtype=torch.long)  # Default lane_labels
        }
