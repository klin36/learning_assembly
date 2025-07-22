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

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

from utils.training_utils import EMAModel
from datasets.pusht_dataset import PushTStateDataset
from models.unet_bc import ConditionalUnet1D


def train_bc(
    zarr_path: str,
    obs_dim: int,
    act_dim: int,
    obs_horizon: int = 2,
    action_horizon: int = 8,
    pred_horizon: int = 16,
    batch_size: int = 256,
    epochs: int = 100,
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
            # Shapes:
            # obs:    (B, obs_horizon, obs_dim)
            # action: (B, pred_horizon, act_dim)

            obs = batch['obs'].to(device)         # (B, obs_horizon, obs_dim)
            action = batch['action'].to(device)   # (B, pred_horizon, act_dim)

            B = obs.shape[0]

            # Flatten observation across time to form global condition
            obs_cond = obs.view(B, -1)             # (B, obs_horizon * obs_dim)

            # Sample noise
            noise = torch.randn_like(action)       # (B, pred_horizon, act_dim)

            # Sample random timestep for each sample in batch
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device
            ).long()

            # Apply forward diffusion: q(x_t | x_0)
            noisy_action = noise_scheduler.add_noise(action, noise, timesteps)

            # Permute to (B, act_dim, pred_horizon) for model
            noisy_action = noisy_action.permute(0, 2, 1)  # (B, act_dim, T)
            noise = noise.permute(0, 2, 1)                # (B, act_dim, T)

            # Predict noise with UNet
            pred_noise = model(noisy_action, timesteps, global_cond=obs_cond)  # (B, act_dim, T)

            # MSE loss between predicted and true noise
            loss = F.mse_loss(pred_noise, noise.permute(0, 2, 1))

            # Backpropagation and optimization
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


    # Save final EMA weights
    os.makedirs("checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"checkpoints/bc_model_{timestamp}.pt"

    ema.copy_to(model.parameters())  # copy EMA weights into model
    torch.save(model.state_dict(), model_path)
    print(f"Saved EMA model to {model_path}")
    wandb.save(model_path)
    wandb.finish()


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
