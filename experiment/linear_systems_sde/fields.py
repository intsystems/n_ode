import torch
from torch import nn
from torchdiffeq import odeint


class LinearField(nn.Module):
    """dx/dt = A x for a fixed matrix A."""

    def __init__(self, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T


def _planar_rotation(theta: float, i: int, j: int) -> torch.Tensor:
    Q = torch.eye(3)
    ct, st = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    Q[i, i], Q[j, j] = ct, ct
    Q[i, j], Q[j, i] = -st, st
    return Q


def matched_spectrum_trio(
    spacing: float = 1.0,
    freq: float = 4.0,
    decay: float = 0.7,
    real_decay: float = 0.6,
) -> list[LinearField]:
    """Three linear systems that SHARE an eigenvalue spectrum but differ in coupling.

    Each system has eigenvalues ``-decay ± i*freq`` (a damped oscillation) and ``-real_decay``
    (a real mode). Because the spectrum is identical, every class produces trajectories with
    the same oscillation frequency and the same amplitude-decay envelope -> matched power
    spectrum and global statistics, so a data-driven LSTM has almost no shape feature to learn.
    The classes differ only by an orthogonal change of basis ``A_k = Q_k B Q_k^T`` (rotation
    angle ``spacing``), i.e. in how the shared modes are mixed across observed coordinates.
    """
    B = torch.tensor([
        [-decay, freq, 0.0],
        [-freq, -decay, 0.0],
        [0.0, 0.0, -real_decay],
    ])
    fields = []
    for theta in (-spacing, 0.0, spacing):
        Q = _planar_rotation(theta, 0, 2) @ _planar_rotation(0.6 * theta, 1, 2)
        fields.append(LinearField(Q @ B @ Q.T))
    return fields


def simulate_sde_paths(
    drift_field: nn.Module, x0: torch.Tensor, t_mesh: torch.Tensor,
    sigma_proc: float,
) -> torch.Tensor:
    """Euler-Maruyama sample paths of ``dx = f(x) dt + sigma_proc * I dW``.

    The drift is propagated one step with the SAME ``odeint`` the UKF ``fx`` uses, so the
    discrete model the classifier assumes -- mean ``odeint(f, x, [0, dt])`` and process
    covariance ``sigma_proc**2 * dt * I`` -- matches the data-generating process exactly.
    Isotropic diffusion means the one-step increment std is ``sigma_proc * sqrt(dt)``.

    Relies on the global RNG (seeded by ``seed_everything``), like the original's
    ``torch.randn_like`` observation noise. Returns ``(T, N, d)``, like ``torchdiffeq.odeint``.
    """
    dt = float(t_mesh[1] - t_mesh[0])
    std = sigma_proc * dt ** 0.5
    x, xs, step_mesh = x0, [x0], torch.tensor([0., dt])
    for _ in range(len(t_mesh) - 1):
        x = odeint(drift_field, x, step_mesh)[-1] + torch.randn_like(x) * std
        xs.append(x)
    return torch.stack(xs, dim=0)
