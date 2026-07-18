import torch
from torch import nn
from torch import optim
from lightning import LightningModule
from torchdiffeq import odeint_adjoint as odeint


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


class Field(nn.Module):
    """Autonomous neural vector field ``dx/dt = f(x)`` for the gyroscope state.

    No control conditioning: the drift depends on the rotation-rate state only,
    so this module is directly usable as the ``f(t, x)`` passed to ``odeint``.
    """

    def __init__(self, d: int):
        super().__init__()
        self.input_dim = d

        LATENT_DIM = 128
        self.transform_to_latent = nn.Linear(self.input_dim, LATENT_DIM)
        self.blocks = nn.Sequential(
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM)
        )
        self.transform_to_orig = nn.Linear(LATENT_DIM, d)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        h = torch.nn.functional.silu(self.transform_to_latent(x))
        h = self.transform_to_orig(self.blocks(h))
        return h


class FieldLitModule(LightningModule):
    def __init__(
        self, d: int, dt: float,
        traj_mean: torch.Tensor, traj_std: torch.Tensor
    ):
        super().__init__()
        self.field = Field(d)
        self.dt = dt
        self.register_buffer("traj_mean", traj_mean.to(torch.float32))
        self.register_buffer("traj_std", traj_std.to(torch.float32))

        self.save_hyperparameters(ignore=["traj_mean", "traj_std"])

    def _rollout(self, traj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # batch: (B, T, d) -> (pred, target) both (T, B, d)
        T = traj.shape[1]
        x0 = traj[:, 0]
        t = torch.arange(T, dtype=torch.float32, device=x0.device) * self.dt

        pred = odeint(self.field, x0, t, method="rk4")
        target = traj.transpose(0, 1)

        return pred, target

    def training_step(self, batch, batch_idx):
        traj = batch
        traj = (traj - self.traj_mean) / self.traj_std
        batch_size = traj.shape[0] * traj.shape[1]

        pred, target = self._rollout(traj)
        loss = nn.functional.mse_loss(pred, target)
        self.log("Train/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        traj = batch
        traj = (traj - self.traj_mean) / self.traj_std
        batch_size = traj.shape[0] * traj.shape[1]

        pred, target = self._rollout(traj)
        loss = nn.functional.mse_loss(pred, target)
        self.log("Val/loss", loss, on_epoch=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.field.parameters(), lr=1e-3, weight_decay=1e-7)