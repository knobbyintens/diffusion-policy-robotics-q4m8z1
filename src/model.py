import torch
import torch.nn as nn


class DiffusionPolicy(nn.Module):
    """Denoising diffusion policy over action chunks."""

    def __init__(self, obs_dim: int, act_dim: int, horizon: int = 16):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        self.denoiser = nn.Sequential(
            nn.Linear(256 + act_dim * horizon, 512), nn.ReLU(),
            nn.Linear(512, act_dim * horizon),
        )

    def forward(self, obs, noisy_actions):
        h = self.encoder(obs)
        return self.denoiser(torch.cat([h, noisy_actions.flatten(1)], dim=-1))
