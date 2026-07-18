import torch
from torch import nn
from torch import optim
from lightning import LightningModule
from torchdiffeq import odeint_adjoint as odeint
from torchcubicspline import NaturalCubicSpline


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


class ControlledField(nn.Module):
    """Neural vector field ``dx/dt = f(x, control)`` for the gyroscope state."""

    def __init__(self, d: int, num_controls: int):
        super().__init__()
        self.input_dim = d + num_controls

        LATENT_DIM = 128
        self.transform_to_latent = nn.Linear(self.input_dim, LATENT_DIM)
        self.blocks = nn.Sequential(
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM)
        )
        self.transform_to_orig = nn.Linear(LATENT_DIM, d)

    def forward(self, t: torch.Tensor, x: torch.Tensor, control: torch.Tensor):
        x = torch.concat((x, control), dim=-1)
        h = torch.nn.functional.silu(self.transform_to_latent(x))
        h = self.transform_to_orig(self.blocks(h))
        return h


class FieldAdapterForControl(nn.Module):
    """Wraps a :class:`ControlledField` into an ``f(t, x)`` for ``odeint``.

    The continuous control (acceleration) is reconstructed in continuous time
    from precomputed natural-cubic-spline coefficients and normalized before it
    is fed to the field.
    """

    def __init__(
        self, controlled_f: ControlledField,
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

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        control = (self.cont_control_spline.evaluate(t) - self.cont_control_mean) / self.cont_control_std
        return self.controlled_f(t, x, control)


class FieldLitModule(LightningModule):
    def __init__(
        self, d: int, num_controls: int, dt: float,
        traj_mean: torch.Tensor, traj_std: torch.Tensor,
        cont_control_mean: torch.Tensor, cont_control_std: torch.Tensor
    ):
        super().__init__()
        self.field = ControlledField(d, num_controls)
        self.dt = dt
        self.register_buffer("traj_mean", traj_mean.to(torch.float32))
        self.register_buffer("traj_std", traj_std.to(torch.float32))
        self.register_buffer("cont_control_mean", cont_control_mean.to(torch.float32))
        self.register_buffer("cont_control_std", cont_control_std.to(torch.float32))

        self.save_hyperparameters(ignore=["traj_mean", "traj_std", "cont_control_mean", "cont_control_std"])

    def _rollout(self, traj: torch.Tensor, spline_coefs) -> tuple[torch.Tensor, torch.Tensor]:
        # batch: (B, T, d) -> (pred, target) both (T, B, d)
        T = traj.shape[1]
        x0 = traj[:, 0]
        t = torch.arange(T, dtype=torch.float32, device=x0.device) * self.dt

        field_adapter = FieldAdapterForControl(
            self.field, spline_coefs,
            self.cont_control_mean, self.cont_control_std,
            self.dt
        )
        pred = odeint(field_adapter, x0, t, method="rk4")
        target = traj.transpose(0, 1)

        return pred, target

    def training_step(self, batch, batch_idx):
        traj = batch[0]
        traj = (traj - self.traj_mean) / self.traj_std
        spline_coefs = batch[1:]
        batch_size = traj.shape[0] * traj.shape[1]

        pred, target = self._rollout(traj, spline_coefs)
        loss = nn.functional.mse_loss(pred, target)
        self.log("Train/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        traj = batch[0]
        traj = (traj - self.traj_mean) / self.traj_std
        spline_coefs = batch[1:]
        batch_size = traj.shape[0] * traj.shape[1]

        pred, target = self._rollout(traj, spline_coefs)
        loss = nn.functional.mse_loss(pred, target)
        self.log("Val/loss", loss, on_epoch=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.field.parameters(), lr=1e-3, weight_decay=1e-7)
