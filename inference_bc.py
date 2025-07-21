import os
import numpy as np
import torch
import collections
from tqdm import tqdm

from envs.pusht import PushTEnv  # Update if you have a different location
from models.unet_bc import ConditionalUnet1D
from utils.ema import EMAModel
from utils.normalization import normalize_data, unnormalize_data
from datasets.pusht_dataset import PushTStateDataset
from diffusers.schedulers import DDPMScheduler

# --- Parameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/bc_model.pt"
zarr_path = "data/pusht/pusht_cchi_v7_replay.zarr"
obs_horizon = 2
pred_horizon = 16
action_horizon = 8
num_diffusion_iters = 100
obs_dim = 5
action_dim = 2
max_steps = 200

# --- Load dataset for stats only ---
dataset = PushTStateDataset(
    dataset_path=zarr_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
stats = dataset.stats

# --- Load model ---
model = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- EMA Model (optional, you can skip this if EMA is not used) ---
# ema = EMAModel(model.parameters(), power=0.75)
# ema.copy_to(model.parameters())

# --- Load scheduler ---
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)
noise_scheduler.set_timesteps(num_diffusion_iters)

# --- Init environment ---
env = PushTEnv()
env.seed(100000)
obs, _ = env.reset()

# Initialize observation history
obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
done = False
step_idx = 0

# --- Main inference loop ---
with tqdm(total=max_steps, desc="Rollout") as pbar:
    while not done:
        # Stack and normalize observations
        obs_seq = np.stack(obs_deque)
        nobs = normalize_data(obs_seq, stats=stats['obs'])
        nobs = torch.tensor(nobs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_horizon, obs_dim)
        obs_cond = nobs.flatten(start_dim=1)  # (1, obs_horizon * obs_dim)

        # Initialize noisy actions
        action = torch.randn((1, pred_horizon, action_dim), device=device)

        # Diffusion process
        for k in noise_scheduler.timesteps:
            with torch.no_grad():
                noise_pred = model(action, k, global_cond=obs_cond)
                action = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=action
                ).prev_sample

        # Unnormalize
        action = action.detach().cpu().numpy()[0]
        action = unnormalize_data(action, stats=stats['action'])

        # Slice predicted actions to execute
        start = obs_horizon - 1
        end = start + action_horizon
        to_execute = action[start:end]

        for u in range(len(to_execute)):
            obs, reward, done, _, _ = env.step(to_execute[u])
            obs_deque.append(obs)

            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(step=step_idx, reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break
