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

        root = zarr.open(os.path.join(zarr_path, "data"), mode='r')
        obs = root[self.obs_key][:]
        act = root[self.action_key][:]

        if obs.ndim == 3:
            obs = obs[:, :7, :]  # use first 7 keypoints only
            obs = obs.reshape(obs.shape[0], -1)
        if act.ndim == 3:
            act = act.reshape(act.shape[0], -1)

        self.obs = obs
        self.action = act

        # Drop samples shorter than horizon
        max_index = len(self.obs) - horizon
        self.valid_idxs = list(range(max_index))

        # Normalization
        if self.use_normalization:
            self.obs_mean = np.mean(self.obs, axis=0)
            self.obs_std = np.std(self.obs, axis=0) + 1e-6
            self.action_mean = np.mean(self.action, axis=0)
            self.action_std = np.std(self.action, axis=0) + 1e-6
        else:
            self.obs_mean = self.obs_std = self.action_mean = self.action_std = None

    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        obs_seq = self.obs[i:i+self.horizon]
        act_seq = self.action[i:i+self.horizon]

        # Normalize
        if self.use_normalization:
            obs_seq = (obs_seq - self.obs_mean) / self.obs_std
            act_seq = (act_seq - self.action_mean) / self.action_std

        # Convert to tensor and transpose to (C, T)
        obs_tensor = torch.tensor(obs_seq, dtype=torch.float32).permute(1, 0)
        act_tensor = torch.tensor(act_seq, dtype=torch.float32).permute(1, 0)

        return {
            "condition": obs_tensor,
            "target": act_tensor
        }

