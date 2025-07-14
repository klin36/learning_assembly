import os
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset

class PushTDataset(Dataset):
    def __init__(self, 
                 zarr_path, 
                 horizon=16,
                 use_normalization=True,
                 obs_key='keypoint',
                 action_key='action'):
        """
        Args:
            zarr_path (str): Path to .zarr folder
            horizon (int): Number of timesteps per sample
            use_normalization (bool): Whether to normalize
            obs_key (str): Observation source ('keypoint', 'state', etc.)
            action_key (str): Action source
        """
        self.zarr_path = zarr_path
        self.horizon = horizon
        self.obs_key = obs_key
        self.action_key = action_key
        self.use_normalization = use_normalization

        # Load all episodes and concatenate
        root = zarr.open(os.path.join(zarr_path, "data"), mode='r')
        self.obs_list = []
        self.action_list = []
        self.episode_starts = []
        self.episode_ends = []

        episode_keys = sorted(root[self.obs_key].keys(), key=lambda x: float(x))
        start_idx = 0

        for k in episode_keys:
            obs = root[self.obs_key][k][:]
            act = root[self.action_key][k][:]

            if len(obs) < self.horizon:
                continue  # skip short episodes

            self.obs_list.append(obs)
            self.action_list.append(act)
            end_idx = start_idx + len(obs)
            self.episode_starts.append(start_idx)
            self.episode_ends.append(end_idx)
            start_idx = end_idx

        self.obs = np.concatenate(self.obs_list, axis=0)
        self.action = np.concatenate(self.action_list, axis=0)

        # Compute normalization stats
        if self.use_normalization:
            self.obs_mean = np.mean(self.obs, axis=0)
            self.obs_std = np.std(self.obs, axis=0) + 1e-6
            self.action_mean = np.mean(self.action, axis=0)
            self.action_std = np.std(self.action, axis=0) + 1e-6
        else:
            self.obs_mean = self.obs_std = self.action_mean = self.action_std = None

        # Compute valid starting indices (avoid crossing episodes)
        self.valid_idxs = []
        for start, end in zip(self.episode_starts, self.episode_ends):
            for i in range(start, end - self.horizon + 1):
                self.valid_idxs.append(i)

    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        obs_seq = self.obs[i:i+self.horizon]
        act_seq = self.action[i:i+self.horizon]

        if self.use_normalization:
            obs_seq = (obs_seq - self.obs_mean) / self.obs_std
            act_seq = (act_seq - self.action_mean) / self.action_std

        obs_tensor = torch.tensor(obs_seq, dtype=torch.float32).T  # (C, T)
        act_tensor = torch.tensor(act_seq, dtype=torch.float32).T  # (C, T)

        return {
            "condition": obs_tensor,
            "target": act_tensor
        }
