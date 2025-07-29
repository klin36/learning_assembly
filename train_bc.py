import os
import math
import collections
from datetime import datetime
from typing import Tuple, Sequence, Dict, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import zarr
import matplotlib.pyplot as plt

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

from utils.training_utils import EMAModel
from models.unet_bc import ConditionalUnet1D
from datasets.pusht_dataset import PushTStateDataset, normalize_data, unnormalize_data

def train_bc(
    zarr_path: str,
    obs_dim: int,
    act_dim: int,
    obs_horizon: int = 2,
    action_horizon: int = 8,
    pred_horizon: int = 16,
    batch_size: int = 256,
    epochs: int = 9000,
    lr: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    project_name: str = 'pusht_bc',
    run_name: Optional[str] = None,
):
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "pred_horizon": pred_horizon,
            "obs_horizon": obs_horizon,
            "action_horizon": action_horizon,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr
        }
    )

    # Load dataset
    dataset = PushTStateDataset(
        dataset_path=zarr_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )

    # Model
    global_cond_dim = obs_dim * obs_horizon
    model = ConditionalUnet1D(
        input_dim=act_dim,
        global_cond_dim=global_cond_dim
    ).to(device)

    # EMA model
    ema = EMAModel(model.parameters(), power=0.75)

    # Diffusion scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # Optimizer and LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * epochs
    )

    # Training loop
    for epoch in tqdm(range(epochs), desc='Epoch'):
        epoch_loss = []
        for batch in tqdm(dataloader, desc='Batch', leave=False):
            obs = batch['obs'].to(device) # (B, obs_horizon, obs_dim)
            action = batch['action'].to(device) # (B, pred_horizon, act_dim)

            B = obs.shape[0]

            # Flatten observation across time to form global condition
            obs_cond = obs.view(B, -1) # (B, obs_horizon * obs_dim)

            # Sample noise
            noise = torch.randn_like(action) # (B, pred_horizon, act_dim)

            # Sample random timestep for each sample in batch
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

            # Apply forward diffusion: q(x_t | x_0)
            noisy_action = noise_scheduler.add_noise(action, noise, timesteps)

            noisy_action = noisy_action.permute(0, 2, 1) # (B, act_dim, T)
            noise = noise.permute(0, 2, 1) # (B, act_dim, T)

            # Predict noise
            pred_noise = model(noisy_action, timesteps, global_cond=obs_cond) # (B, act_dim, T)

            # MSE loss
            loss = F.mse_loss(pred_noise, noise.permute(0, 2, 1))

            # Backprop and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            ema.step(model.parameters())

            epoch_loss.append(loss.item())

        # Logging
        avg_loss = np.mean(epoch_loss)
        print(f"[Epoch {epoch+1}] Avg loss: {avg_loss:.6f}")
        wandb.log({"loss": avg_loss, "epoch": epoch + 1})

        if (epoch + 1) % 500 == 0:
            visualize_inference(model, dataset, dataloader, noise_scheduler, dataset.stats, device, epoch, obs_horizon)

    # Save final EMA weights
    os.makedirs("checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"checkpoints/bc_model_{timestamp}.pt"
    ema.copy_to(model.parameters())
    torch.save(model.state_dict(), model_path)
    print(f"Saved EMA model to {model_path}")
    wandb.save(model_path)
    wandb.finish()


def visualize_inference(
    model, dataset, dataloader, noise_scheduler, stats, device, epoch, obs_horizon, 
    save_path_prefix="training_vis/inference_epoch", wandb_log=True):

    model.eval()

    sample = next(iter(dataloader))
    obs = sample['obs'][:1].to(device) # (1, obs_dim, obs_horizon)
    gt_action = sample['action'][:1].to(device) # (1, act_dim, pred_horizon)
    obs_cond = obs.view(1, -1) # (1, obs_dim * obs_horizon)

    # Initialize noise and denoise
    x = torch.randn((1, gt_action.shape[-1], gt_action.shape[1]), device=device) # (1, T, act_dim)
    for t in noise_scheduler.timesteps:
        model_input = x.permute(0, 2, 1) # (B, act_dim, T)
        with torch.no_grad():
            noise_pred = model(model_input, t, obs_cond)
        noise_pred = noise_pred.permute(0, 2, 1)
        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    # Convert and unnormalize
    pred_action = x.detach().cpu().numpy()[0] # (T, act_dim)
    gt_action_np = gt_action.cpu().numpy()[0].T # (T, act_dim)
    pred_action = unnormalize_data(pred_action, stats['action'])
    gt_action_np = unnormalize_data(gt_action_np, stats['action'])

    # Plot trajectories
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gt_action_np[:, 0], gt_action_np[:, 1], label='Ground Truth', color='blue')
    ax.plot(pred_action[:, 0], pred_action[:, 1], label='Predicted', color='red')
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect('equal')
    ax.set_title(f"2D Action Trajectory (Epoch {epoch+1})")
    ax.legend()
    plt.tight_layout()

    image_path = f"{save_path_prefix}_{epoch+1}.png"
    # plt.savefig(image_path)
    plt.close(fig)

    if wandb_log:
        wandb.log({"inference_plots": [wandb.Image(image_path, caption=f"Epoch {epoch+1}")]})

    model.train()


if __name__ == "__main__":
    train_bc(
        zarr_path="data/pusht/pusht_cchi_v7_replay.zarr",
        obs_dim=5,
        act_dim=2,
        obs_horizon=2,
        action_horizon=8,
        pred_horizon=16,
        run_name="bc_unet1d_ema"
    )
