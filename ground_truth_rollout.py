import os
import numpy as np
import zarr
import torch
from tqdm import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from envs.pusht_env import PushTEnv

# Parameters
zarr_path = "data/pusht/pusht_cchi_v7_replay.zarr"
render_fps = 10
video_path = "ground_truth_rollout.mp4"
episode_idx = 3  # change to visualize a different episode

# Load full Zarr data
dataset_root = zarr.open(zarr_path, 'r')
all_actions = dataset_root['data']['action'][:]  # shape (N, action_dim)
episode_ends = dataset_root['meta']['episode_ends'][:]

statess = dataset_root['data']['state'][:]

# Determine episode bounds
start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
end_idx = episode_ends[episode_idx]
actions = all_actions[start_idx:end_idx] # don't unnormalize

# Rollout in env
env = PushTEnv()
env.seed(0)
obs, _ = env.reset()
env._set_state(statess[start_idx])


imgs = [env.render(mode="rgb_array")]

for t, action in enumerate(actions):
    print(f"[{t}] Action: {action}, Norm: {np.linalg.norm(action):.4f}")
    obs, reward, done, _, _ = env.step(action)
    imgs.append(env.render(mode="rgb_array"))

# Save video
clip = ImageSequenceClip(imgs, fps=render_fps)
clip.write_videofile(video_path, codec="libx264")
print(f"Saved ground truth video to {video_path}")
