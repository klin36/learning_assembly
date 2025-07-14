import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from diffusers.schedulers import DDPMScheduler

from datasets.pusht_dataset import PushTDataset
from models.unet_bc import ConditionalUNet1D

def train_bc(
    zarr_path,
    obs_dim,
    act_dim,
    horizon=16,
    batch_size=64,
    epochs=50,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_path='checkpoints/bc_model.pt',
    project_name='pusht_bc',
    run_name=None,
):

    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "horizon": horizon,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "scheduler": "DDPM",
            "dataset": os.path.basename(zarr_path),
        }
    )

    dataset = PushTDataset(
        zarr_path=zarr_path,
        horizon=horizon,
        obs_key='keypoint',
        action_key='action'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConditionalUNet1D(obs_dim, act_dim, horizon).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            condition = batch['condition'].to(device)   # (B, obs_dim, T)
            target = batch['target'].to(device)         # (B, act_dim, T)

            # Sample random timestep for each batch element
            bsz = target.shape[0]
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

            # Add noise to target action
            noise = torch.randn_like(target)
            noisy_target = noise_scheduler.add_noise(original_samples=target, noise=noise, timesteps=t)

            # Predict noise
            pred = model(noisy_target, t, condition)  # (B, act_dim, T)

            # MSE loss
            loss = F.mse_loss(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Avg loss: {avg_loss:.6f}")
        wandb.log({"loss": avg_loss, "epoch": epoch + 1})

        # Save checkpoint
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")
        wandb.save(save_path)

    wandb.finish()

if __name__ == "__main__":
    train_bc(
        zarr_path="data/pusht/pusht_cchi_v7_replay.zarr",
        obs_dim=14,
        act_dim=2,
        horizon=16,
        run_name="bc_unet1d_pusht"
    )