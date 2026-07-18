import math

import torch
from torch import nn
from torch import optim
from lightning import LightningModule


def euler_maruyama_transition(
    field: "SDEField", t: torch.Tensor, traj: torch.Tensor,
    dt: float, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-step Euler-Maruyama predictive mean and variance for the autonomous
    SDE ``dx = f(t, x) dt + g(t, x) dW`` with diagonal noise (no control).

    Given the observed states ``x_n = traj[..., n, :]`` the Euler-Maruyama
    discretization implies the Gaussian transition

        x_{n+1} | x_n ~ N(x_n + f(t_n, x_n) dt,  diag(g(t_n, x_n)^2 dt)).

    Returns
    -------
    (mean, var) : predictive mean and variance, both ``(..., T - 1, d)``,
        for the transitions from step ``n`` to step ``n + 1``.
    """
    f = field.f(t, traj)
    g = field.g(t, traj)
    mean = traj[..., :-1, :] + f[..., :-1, :] * dt
    var = g[..., :-1, :] ** 2 * dt + eps
    return mean, var


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Elementwise negative log-likelihood of a diagonal Gaussian."""
    return 0.5 * ((target - mean) ** 2 / var + torch.log(2 * math.pi * var))


class ResidualBlock(nn.Module):
    def __init__(self, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden)
        )

    def forward(self, x):
        return x + self.net(x)


class SDEField(nn.Module):
    """Autonomous drift ``f(x)`` and constant diagonal diffusion ``g`` for the
    gyroscope state SDE.

    No control conditioning: the drift depends on the rotation-rate state only,
    and the diffusion is a state- and time-independent learnable per-channel
    sigma (the controlled variant conditioned the diffusion on the acceleration,
    which no longer exists here).
    """

    sde_type = "stratonovich"
    noise_type = "diagonal"

    def __init__(self, d: int):
        super().__init__()
        self.input_dim = d

        LATENT_DIM = 30
        self.transform_to_latent = nn.Linear(self.input_dim, LATENT_DIM)
        self.blocks = nn.Sequential(
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM)
        )
        self.transform_to_orig = nn.Linear(LATENT_DIM, d)

        # constant (state- and time-independent) learnable diagonal diffusion
        self.brownian_sigma = nn.Parameter(torch.ones(d))

    def f(self, t: torch.Tensor, x: torch.Tensor):
        h = torch.nn.functional.silu(self.transform_to_latent(x))
        h = self.transform_to_orig(self.blocks(h))
        return h

    def g(self, t: torch.Tensor, x: torch.Tensor):
        # broadcast the per-channel constant diffusion to the shape of x
        return self.brownian_sigma.expand_as(x)


class FieldLitModule(LightningModule):
    def __init__(
        self, d: int, dt: float,
        traj_mean: torch.Tensor, traj_std: torch.Tensor
    ):
        super().__init__()
        self.field = SDEField(d)
        self.dt = dt
        self.register_buffer("traj_mean", traj_mean.to(torch.float32))
        self.register_buffer("traj_std", traj_std.to(torch.float32))

        self.save_hyperparameters(ignore=["traj_mean", "traj_std"])

    def predictive_transition(self, traj_norm: torch.Tensor):
        """Euler-Maruyama one-step predictive mean/variance for a normalized trajectory."""
        T = traj_norm.shape[-2]
        t = torch.arange(T, dtype=torch.float32, device=traj_norm.device) * self.dt
        return euler_maruyama_transition(self.field, t, traj_norm, self.dt)

    def transition_nll(self, traj_norm: torch.Tensor) -> torch.Tensor:
        """Mean Gaussian NLL of a normalized trajectory under the Euler-Maruyama transition."""
        mean, var = self.predictive_transition(traj_norm)
        target = traj_norm[..., 1:, :]
        # sum over state dims (joint diagonal Gaussian), average over transitions/batch
        return gaussian_nll(target, mean, var).sum(-1).mean()

    def training_step(self, batch, batch_idx):
        traj = batch
        traj = (traj - self.traj_mean) / self.traj_std
        batch_size = traj.shape[0] * traj.shape[1]

        loss = self.transition_nll(traj)
        self.log("Train/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        traj = batch
        traj = (traj - self.traj_mean) / self.traj_std
        batch_size = traj.shape[0] * traj.shape[1]

        loss = self.transition_nll(traj)
        self.log("Val/loss", loss, on_epoch=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.field.parameters(), lr=1e-3, weight_decay=1e-9)
