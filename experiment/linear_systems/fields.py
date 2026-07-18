import torch
from torch import nn


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
    """Three linear systems that share an eigenvalue spectrum but differ in coupling.

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
