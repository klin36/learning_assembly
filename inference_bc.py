import os
import numpy as np
import torch
import collections
from tqdm import tqdm
import zarr
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from envs.pusht_env import PushTEnv
from models.unet_bc import ConditionalUnet1D
from utils.training_utils import EMAModel
from datasets.pusht_dataset import PushTStateDataset, normalize_data, unnormalize_data
from diffusers.schedulers import DDPMScheduler, DDIMScheduler

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/bc_model_10000.pt"
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

# Dataset conditioning code
dataset_root = zarr.open(zarr_path, 'r')
episode_ends = dataset_root['meta']['episode_ends'][:]
states = dataset_root['data']['state'][:]
episode_idx = 3
offset = 2
start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
t0 = start_idx + offset

# Model
model = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Scheduler
noise_scheduler = DDIMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)
noise_scheduler.set_timesteps(25)

# Environment
env = PushTEnv()
env.seed(10000)
obs, _ = env.reset()
env._set_state(states[t0]) # comment out to try random init
obs = states[t0] # comment out to try random init

# Observation history
obs_deque = collections.deque([states[t0 - i] for i in reversed(range(obs_horizon))], maxlen=obs_horizon) # comment out to try random init
# obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
print("Initial env state:", states[t0])
print("Observation history:")
for i, o in enumerate(obs_deque):
    print(f"  t-{obs_horizon - 1 - i}: {o}")
done = False
step_idx = 0
imgs = [env.render(mode='rgb_array')]  # store first frame for video

# Main inference loop
with tqdm(total=max_steps, desc="Inference") as pbar:
    while not done:
        # Stack and normalize observations
        obs_seq = np.stack(obs_deque)
        nobs = normalize_data(obs_seq, stats=stats['obs'])
        nobs = torch.tensor(nobs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_horizon, obs_dim)
        obs_cond = nobs.flatten(start_dim=1)  # (1, obs_horizon * obs_dim)

        # Initialize noisy actions
        naction = torch.randn((1, pred_horizon, action_dim), device=device)

        # Diffusion
        with torch.no_grad():
            for k in noise_scheduler.timesteps:
                noise_pred = model(naction, timestep=k, global_cond=obs_cond)
                noise_pred = noise_pred.permute(0, 2, 1) # SHOULD I??
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # Unnormalize
        naction = naction.detach().cpu().numpy()[0]
        action_pred = unnormalize_data(naction, stats=stats['action'])

        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end, :] # maybe get rid of second indexing
        print("Predicted action:", action)
        print("Raw predicted action (normalized):", naction)

        for u in range(len(action)):
            obs, reward, done, _, _ = env.step(action[u])
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
