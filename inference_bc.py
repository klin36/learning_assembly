import os
import numpy as np
import torch
import collections
from tqdm import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from envs.pusht_env import PushTEnv
from models.unet_bc import ConditionalUnet1D
from utils.training_utils import EMAModel
from datasets.pusht_dataset import PushTStateDataset, normalize_data, unnormalize_data
from diffusers.schedulers import DDPMScheduler # TODO SWITCH TO DDIM

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/bc_model_2025-07-23_11-47-42.pt"
zarr_path = "data/pusht/pusht_cchi_v7_replay.zarr"
obs_horizon = 2
pred_horizon = 16
action_horizon = 8
num_diffusion_iters = 100
obs_dim = 5
action_dim = 2
max_steps = 200

# Load dataset for stats only
dataset = PushTStateDataset(
    dataset_path=zarr_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
stats = dataset.stats

# Model
model = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Scheduler
noise_scheduler = DDPMScheduler( # TODO SWITCH TO DDIM
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)
noise_scheduler.set_timesteps(num_diffusion_iters)

# Environment
env = PushTEnv()
env.seed(100000)
obs, _ = env.reset()

# Observation history
obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
done = False
step_idx = 0

imgs = [env.render(mode='rgb_array')]  # store first frame for video

# Main inference loop
with tqdm(total=max_steps, desc="Rollout") as pbar:
    while not done:
        # Stack and normalize observations
        obs_seq = np.stack(obs_deque)
        nobs = normalize_data(obs_seq, stats=stats['obs'])
        nobs = torch.tensor(nobs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_horizon, obs_dim)
        obs_cond = nobs.flatten(start_dim=1)  # (1, obs_horizon * obs_dim)

        # Initialize noisy actions
        action = torch.randn((1, pred_horizon, action_dim), device=device)

        # Diffusion
        with torch.no_grad():
            for k in noise_scheduler.timesteps:
                noise_pred = model(action, k, global_cond=obs_cond)
                noise_pred = noise_pred.permute(0, 2, 1)
                action = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=action
                ).prev_sample

        # Unnormalize
        action = action.detach().cpu().numpy()[0]
        action = unnormalize_data(action, stats=stats['action'])

        start = obs_horizon - 1
        end = start + action_horizon
        to_execute = action[start:end]
        # print("Predicted action:", to_execute)
        # print("Raw predicted action (normalized):", action)
        # print("Unnormalized:", unnormalize_data(action, stats['action']))


        for u in range(len(to_execute)):
            obs, reward, done, _, _ = env.step(to_execute[u])
            obs_deque.append(obs)
            imgs.append(env.render(mode='rgb_array')) # adding frames to video

            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(step=step_idx, reward=reward)
            if step_idx >= max_steps:
                done = True
            if done:
                break

        
video_path = "inference.mp4"
clip = ImageSequenceClip(imgs, fps=10)
clip.write_videofile(video_path, codec="libx264")
print(f"Saved video to {video_path}")
