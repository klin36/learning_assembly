import torch
from diffusers.schedulers import DDIMScheduler
import wandb
import matplotlib.pyplot as plt
import numpy as np

from datasets.pusht_dataset import PushTDataset
from models.unet_bc import ConditionalUNet1D

@torch.no_grad()
def run_inference(
    model_path='checkpoints/bc_model.pt',
    zarr_path='data/pusht/pusht_cchi_v7_replay.zarr',
    obs_dim=14,
    act_dim=2,
    horizon=32,
    num_steps=50, # inference steps
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # dataset for conditioning only
    dataset = PushTDataset(
        zarr_path=zarr_path,
        horizon=horizon,
        obs_key='keypoint',
        action_key='action',
    )

    # REPLACE LATER with some validation logic
    sample = dataset[0]
    condition = sample['condition'].unsqueeze(0).to(device)  # (1, obs_dim, T)

    model = ConditionalUNet1D(obs_dim, act_dim, horizon).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
        set_alpha_to_one=False
    )

    noisy_action = torch.randn(1, act_dim, horizon).to(device)

    scheduler.set_timesteps(num_steps)
    for t in scheduler.timesteps:
        model_output = model(noisy_action, t, condition)

        noisy_action = scheduler.step(
            model_output=model_output,
            timestep=t,
            sample=noisy_action
        ).prev_sample

    action_pred = noisy_action.squeeze(0).permute(1, 0).cpu().numpy()
    action_pred = action_pred * dataset.action_std + dataset.action_mean

    print(f"Sampled action trajectory (shape {action_pred.shape}):")
    print(action_pred)

    # Denormalize ground truth actions
    target_actions = sample['target'].T.cpu()
    target_actions = target_actions.numpy()
    target_actions = target_actions * dataset.action_std + dataset.action_mean

    # Plot predictions vs ground truth
    timesteps = list(range(action_pred.shape[0]))
    plt.figure(figsize=(8, 4))

    plt.plot(timesteps, action_pred[:, 0], label='pred_dx', color='blue')
    plt.plot(timesteps, target_actions[:, 0], label='gt_dx', color='blue', linestyle='--')

    plt.plot(timesteps, action_pred[:, 1], label='pred_dy', color='red')
    plt.plot(timesteps, target_actions[:, 1], label='gt_dy', color='red', linestyle='--')

    plt.title("Predicted vs Ground Truth Actions")
    plt.xlabel("Timestep")
    plt.ylabel("Delta (dx, dy)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("inference_plot.png")
    plt.show()

    return action_pred

if __name__ == "__main__":
    run_inference(model_path='checkpoints/bc_model.pt')

    


