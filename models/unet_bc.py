import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        
        def get_compatible_groupnorm(num_channels, max_groups=8):
            for g in reversed(range(1, max_groups + 1)):
                if num_channels % g == 0:
                    return nn.GroupNorm(g, num_channels)
            raise ValueError(f"No valid group number for {num_channels}")

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            get_compatible_groupnorm(out_channels, max_groups=n_groups),
            nn.Mish()
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1))  # (B, C, 1)
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)

        cond_embed = self.cond_encoder(cond)
        cond_embed = cond_embed.view(cond.shape[0], 2, self.out_channels, 1)
        scale = cond_embed[:, 0]
        bias = cond_embed[:, 1]
        out = scale * out + bias

        out = self.blocks[1](out)
        return out + self.residual_conv(x)


# class ConditionalUnet1D(nn.Module):
#     def __init__(self,
#                  input_dim,
#                  global_cond_dim,
#                  diffusion_step_embed_dim=256,
#                  down_dims=[256, 512, 1024],
#                  kernel_size=5,
#                  n_groups=8):
#         super().__init__()

#         self.input_dim = input_dim
#         cond_dim = diffusion_step_embed_dim + global_cond_dim

#         self.diffusion_step_encoder = nn.Sequential(
#             SinusoidalPosEmb(diffusion_step_embed_dim),
#             nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
#             nn.Mish(),
#             nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim)
#         )

#         self.input_proj = Conv1dBlock(input_dim, down_dims[0], kernel_size, n_groups)

#         # Downsampling path
#         self.down_modules = nn.ModuleList()
#         for i in range(len(down_dims) - 1):
#             dim_in, dim_out = down_dims[i], down_dims[i + 1]
#             self.down_modules.append(nn.ModuleList([
#                 ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
#                 ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
#                 Downsample1d(dim_out)
#             ]))
#         # Final down block (no downsample)
#         self.down_modules.append(nn.ModuleList([
#             ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], cond_dim, kernel_size, n_groups),
#             ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], cond_dim, kernel_size, n_groups),
#             nn.Identity()
#         ]))

#         # Middle blocks
#         mid_dim = down_dims[-1]
#         self.mid_modules = nn.ModuleList([
#             ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
#             ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups)
#         ])

#         # Upsampling path
#         self.up_modules = nn.ModuleList()
#         reversed_dims = list(reversed(down_dims))          # [1024, 512, 256]
#         skip_dims = list(reversed(down_dims[:-1]))         # [512, 256]

#         for i in range(len(skip_dims)):
#             dim_in = reversed_dims[i]
#             skip_dim = skip_dims[i]
#             dim_out = reversed_dims[i + 1]
#             in_channels = dim_in + skip_dim

#             self.up_modules.append(nn.ModuleList([
#                 ConditionalResidualBlock1D(in_channels, dim_out, cond_dim, kernel_size, n_groups),
#                 ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
#                 Upsample1d(dim_out)
#             ]))

#         # Final up block (no skip, no upsample)
#         self.up_modules.append(nn.ModuleList([
#             ConditionalResidualBlock1D(down_dims[0], down_dims[0], cond_dim, kernel_size, n_groups),
#             ConditionalResidualBlock1D(down_dims[0], down_dims[0], cond_dim, kernel_size, n_groups),
#             nn.Identity()
#         ]))

#         self.final_conv = nn.Sequential(
#             Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
#             nn.Conv1d(down_dims[0], input_dim, kernel_size=1)
#         )

#         print("Number of parameters: {:e}".format(sum(p.numel() for p in self.parameters())))

#     def forward(self, x: torch.Tensor, timestep: Union[int, torch.Tensor], global_cond: torch.Tensor):
#         """
#         Args:
#             x: (B, T, input_dim) or (B, input_dim, T)
#             timestep: (B,) or scalar
#             global_cond: (B, cond_dim)
#         Returns:
#             x: (B, input_dim, T)
#         """
#         if x.ndim == 3 and x.shape[1] != self.input_dim:
#             x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)

#         x = self.input_proj(x)

#         # Encode timestep
#         if not torch.is_tensor(timestep):
#             timestep = torch.tensor([timestep], device=x.device)
#         if timestep.ndim == 0:
#             timestep = timestep[None].to(x.device)
#         timestep = timestep.expand(x.shape[0])
#         cond = torch.cat([self.diffusion_step_encoder(timestep), global_cond], dim=-1)

#         # Downsampling path
#         skips = []
#         for i, (res1, res2, down) in enumerate(self.down_modules):
#             x = res1(x, cond)
#             x = res2(x, cond)
#             if i < len(self.down_modules) - 1:
#                 skips.append(x)  # Only save skip if downsampling follows
#             x = down(x)

#         # Mid blocks
#         for block in self.mid_modules:
#             x = block(x, cond)

#         # Upsampling path
#         for i, (res1, res2, up) in enumerate(self.up_modules):
#             if i < len(skips):
#                 skip = skips.pop()
#                 print(f"[UP BLOCK {i}] x shape: {x.shape}, skip shape: {skip.shape}")
#                 if x.shape[-1] != skip.shape[-1]:
#                     print(f"[WARNING] x.shape={x.shape}, skip.shape={skip.shape} — resizing x to match skip")
#                     x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
#                 x = torch.cat([x, skip], dim=1)

#             x = res1(x, cond)
#             x = res2(x, cond)
#             x = up(x)

#         return self.final_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=64):
        super().__init__()

        cond_dim = diffusion_step_embed_dim + global_cond_dim
        hidden_dim = 128
        self.input_dim = input_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim)
        )

        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)

        # Down block
        self.down1 = ConditionalResidualBlock1D(hidden_dim, hidden_dim, cond_dim)

        # Mid block
        self.mid = ConditionalResidualBlock1D(hidden_dim, hidden_dim, cond_dim)

        # Up block (takes skip connection, so 128+128)
        self.up1 = ConditionalResidualBlock1D(hidden_dim * 2, hidden_dim, cond_dim)

        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, timestep: Union[int, torch.Tensor], global_cond: torch.Tensor):
        if x.ndim == 3 and x.shape[1] != self.input_dim:
            x = x.permute(0, 2, 1)  # (B, T, C) → (B, C, T)

        # Encode timestep
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=x.device)
        if timestep.ndim == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])
        cond = torch.cat([self.diffusion_step_encoder(timestep), global_cond], dim=-1)

        # Down path
        x = self.input_proj(x)
        skip = self.down1(x, cond)

        # Mid
        x = self.mid(skip, cond)

        # Up path (concat skip)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.up1(x, cond)

        return self.output_proj(x)
