import math
from itertools import chain

import torch
from torch import nn
from torch import optim
from lightning import LightningModule
from torchcubicspline import NaturalCubicSpline


def euler_maruyama_transition(
    field: "SDEField", t: torch.Tensor, traj: torch.Tensor, control: torch.Tensor,
    dt: float, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-step Euler-Maruyama predictive mean and variance for the SDE
    ``dx = f(t, x) dt + g(t, x) dW`` with diagonal noise.

    Given the observed states ``x_n = traj[..., n, :]`` the Euler-Maruyama
    discretization implies the Gaussian transition

        x_{n+1} | x_n ~ N(x_n + f(t_n, x_n) dt,  diag(g(t_n, x_n)^2 dt)).

    Returns
    -------
    (mean, var) : predictive mean and variance, both ``(..., T - 1, d)``,
        for the transitions from step ``n`` to step ``n + 1``.
    """
    f = field.f(t, traj, control)
    g = field.g(t, traj, control)
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
    """Drift ``f`` and diagonal diffusion ``g`` for the gyroscope state SDE."""

    sde_type = "stratonovich"
    noise_type = "diagonal"

    def __init__(self, d: int, num_controls: int):
        super().__init__()
        self.input_dim = d + num_controls

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

        self.transform_to_latent_sigma = nn.Linear(num_controls, LATENT_DIM)
        self.blocks_sigma = nn.Sequential(
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM)
        )
        self.transform_to_orig_sigma = nn.Linear(LATENT_DIM, d)

    def f(self, t: torch.Tensor, x: torch.Tensor, control: torch.Tensor):
        h = torch.concat((x, control), dim=-1)
        h = torch.nn.functional.silu(self.transform_to_latent(h))
        h = self.transform_to_orig(self.blocks(h))
        return h

    def g(self, t: torch.Tensor, x: torch.Tensor, control: torch.Tensor):
        h = torch.nn.functional.silu(self.transform_to_latent_sigma(control))
        h = self.transform_to_orig_sigma(self.blocks_sigma(h))
        return h


class FieldAdapterForControl(nn.Module):
    """Wraps an :class:`SDEField` into ``f(t, x)`` / ``g(t, x)`` for filtering.

    The continuous control (acceleration) is reconstructed in continuous time
    from precomputed natural-cubic-spline coefficients and normalized before it
    is fed to the field.
    """

    sde_type = "stratonovich"
    noise_type = "diagonal"

    def __init__(
        self, controlled_f: SDEField,
        cont_controls_spline_coefs: list[torch.Tensor],   # already batched
        cont_control_mean: torch.Tensor, cont_control_std: torch.Tensor,
        dt: float = 0.1
    ):
        super().__init__()
        self.controlled_f = controlled_f

        traj_len = cont_controls_spline_coefs[0].shape[-2] + 1
        self.cont_control_spline = NaturalCubicSpline(
            [torch.arange(traj_len, dtype=torch.float32, device=cont_control_mean.device) * dt]
            + list(cont_controls_spline_coefs)
        )
        self.cont_control_mean = cont_control_mean
        self.cont_control_std = cont_control_std

    def f(self, t: torch.Tensor, x: torch.Tensor):
        control = (self.cont_control_spline.evaluate(t) - self.cont_control_mean) / self.cont_control_std
        return self.controlled_f.f(t, x, control)

    def g(self, t: torch.Tensor, x: torch.Tensor):
        control = (self.cont_control_spline.evaluate(t) - self.cont_control_mean) / self.cont_control_std
        return self.controlled_f.g(t, x, control)


class FieldLitModule(LightningModule):
    def __init__(
        self, d: int, num_controls: int, dt: float,
        traj_mean: torch.Tensor, traj_std: torch.Tensor,
        cont_control_mean: torch.Tensor, cont_control_std: torch.Tensor
    ):
        super().__init__()
        self.field = SDEField(d, num_controls)
        self.dt = dt
        self.register_buffer("traj_mean", traj_mean.to(torch.float32))
        self.register_buffer("traj_std", traj_std.to(torch.float32))
        self.register_buffer("cont_control_mean", cont_control_mean.to(torch.float32))
        self.register_buffer("cont_control_std", cont_control_std.to(torch.float32))

        self.save_hyperparameters(ignore=["traj_mean", "traj_std", "cont_control_mean", "cont_control_std"])

    def _build_control(self, T: int, spline_coefs, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Time mesh and (normalized) control aligned with a ``(..., T, d)`` trajectory.

        Works both for batched windows (``spline_coefs`` of shape ``(B, T - 1, C)``)
        and for a single unbatched trajectory (``(T - 1, C)``).
        """
        t = torch.arange(T, dtype=torch.float32, device=device) * self.dt

        traj_len = spline_coefs[0].shape[-2] + 1
        cont_control_spline = NaturalCubicSpline(
            [torch.arange(traj_len, dtype=torch.float32, device=device) * self.dt] + list(spline_coefs)
        )
        control = (cont_control_spline.evaluate(t) - self.cont_control_mean) / self.cont_control_std
        return t, control

    def predictive_transition(self, traj_norm: torch.Tensor, spline_coefs):
        """Euler-Maruyama one-step predictive mean/variance for a normalized trajectory."""
        T = traj_norm.shape[-2]
        t, control = self._build_control(T, spline_coefs, traj_norm.device)
        return euler_maruyama_transition(self.field, t, traj_norm, control, self.dt)

    def transition_nll(self, traj_norm: torch.Tensor, spline_coefs) -> torch.Tensor:
        """Mean Gaussian NLL of a normalized trajectory under the Euler-Maruyama transition."""
        mean, var = self.predictive_transition(traj_norm, spline_coefs)
        target = traj_norm[..., 1:, :]
        # sum over state dims (joint diagonal Gaussian), average over transitions/batch
        return gaussian_nll(target, mean, var).sum(-1).mean()

    def training_step(self, batch, batch_idx):
        traj = batch[0]
        traj = (traj - self.traj_mean) / self.traj_std
        spline_coefs = batch[1:]
        batch_size = traj.shape[0] * traj.shape[1]

        loss = self.transition_nll(traj, spline_coefs)
        self.log("Train/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        traj = batch[0]
        traj = (traj - self.traj_mean) / self.traj_std
        spline_coefs = batch[1:]
        batch_size = traj.shape[0] * traj.shape[1]

        loss = self.transition_nll(traj, spline_coefs)
        self.log("Val/loss", loss, on_epoch=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.field.parameters(), lr=1e-3, weight_decay=1e-9)
