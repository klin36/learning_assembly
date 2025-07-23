import os
import numpy as np
import torch
from tqdm import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from envs.pusht_env import PushTEnv
from datasets.pusht_dataset import PushTStateDataset
from datasets.pusht_dataset import unnormalize_data

# Parameters
zarr_path = "data/pusht/pusht_cchi_v7_replay.zarr"
obs_horizon = 2
action_horizon = 8
pred_horizon = 16
render_fps = 10
video_path = "ground_truth_rollout.mp4"

# Load dataset
dataset = PushTStateDataset(
    dataset_path=zarr_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
sample = dataset[0]
obs_seq = sample["obs"].permute(1, 0).numpy()      # (obs_horizon, obs_dim)
# action_seq = sample["action"].permute(1, 0).numpy()  # (action_horizon, act_dim)

action_seq = sample["action"].permute(1, 0).numpy()
action_seq = unnormalize_data(action_seq, dataset.stats["action"])

# Rollout in env
env = PushTEnv()
env.seed(0)  # use same seed as training if needed
obs, _ = env.reset()
imgs = [env.render(mode="rgb_array")]

for action in action_seq:
    obs, reward, done, _, _ = env.step(action)
    imgs.append(env.render(mode="rgb_array"))

# Save video
clip = ImageSequenceClip(imgs, fps=render_fps)
clip.write_videofile(video_path, codec="libx264")
print(f"Saved ground truth video to {video_path}")