import os
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset

def create_sample_indices(episode_ends, sequence_length, pad_before=0, pad_after=0):
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx
            ])
    return np.array(indices)


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros((sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    return {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }


def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'] + 1e-6)
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    return ndata * (stats['max'] - stats['min']) + stats['min']


class PushTStateDataset(Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):
        dataset_root = zarr.open(dataset_path, 'r')
        train_data = {
            'action': dataset_root['data']['action'][:],
            'obs': dataset_root['data']['state'][:],
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]

        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1
        )

        stats = dict()
        normalized_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_data = normalized_data
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        nsample = sample_sequence(
            train_data=self.normalized_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        obs = nsample['obs'][:self.obs_horizon, :]
        action = nsample['action'][:self.action_horizon, :]

        obs = torch.tensor(obs, dtype=torch.float32).permute(1, 0)     # (obs_dim, T)
        action = torch.tensor(action, dtype=torch.float32).permute(1, 0)  # (act_dim, T)

        return {
            'obs': obs,
            'action': action
        }
