import torch
import torch.nn as nn
from diffusers.models import UNet1DModel

class ConditionalUNet1D(nn.Module):
    def __init__(self, obs_dim, act_dim, horizon, model_channels=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon

        self.in_channels = obs_dim + act_dim
        self.out_channels = act_dim

        self.net = UNet1DModel(
            sample_size=horizon,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            layers_per_block=2,
            block_out_channels=(256, 512), # (model_channels, model_channels * 2)
            down_block_types=("DownBlock1D", "DownBlock1D"),
            up_block_types=("UpBlock1D", "UpBlock1D"),
            mid_block_type="UNetMidBlock1D"
        )

    def forward(self, noisy_action, timestep, condition):
        """
        Args:
            noisy_action: (B, act_dim, T)
            timestep: (B,) or scalar
            condition: (B, obs_dim, T)
        Returns:
            pred noise: (B, act_dim, T)
        """
        # Concatenate along channel dim: [obs | noisy_action]
        x = torch.cat([condition, noisy_action], dim=1)
        return self.net(x, timestep).sample
