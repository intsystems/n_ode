import torch
from torch import nn


class LorenzField(nn.Module):
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
        super().__init__()
        self.register_buffer("sigma", torch.tensor(sigma))
        self.register_buffer("rho", torch.tensor(rho))
        self.register_buffer("beta", torch.tensor(beta))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = x.unbind(-1)
        dx1 = self.sigma * (x2 - x1)
        dx2 = x1 * (self.rho - x3) - x2
        dx3 = x1 * x2 - self.beta * x3
        return torch.stack([dx1, dx2, dx3], dim=-1)


class RosslerField(nn.Module):
    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        super().__init__()
        self.register_buffer("a", torch.tensor(a))
        self.register_buffer("b", torch.tensor(b))
        self.register_buffer("c", torch.tensor(c))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = x.unbind(-1)
        dx1 = -x2 - x3
        dx2 = x1 + self.a * x2
        dx3 = self.b + x3 * (x1 - self.c)
        return torch.stack([dx1, dx2, dx3], dim=-1)


class ChuaField(nn.Module):
    def __init__(
        self,
        alpha: float = 15.6,
        beta: float = 28.0,
        m0: float = -1.143,
        m1: float = -0.714,
    ):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("m0", torch.tensor(m0))
        self.register_buffer("m1", torch.tensor(m1))

    def _diode(self, x: torch.Tensor) -> torch.Tensor:
        return self.m1 * x + 0.5 * (self.m0 - self.m1) * (
            torch.abs(x + 1.0) - torch.abs(x - 1.0)
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = x.unbind(-1)
        dx1 = self.alpha * (x2 - x1 - self._diode(x1))
        dx2 = x1 - x2 + x3
        dx3 = -self.beta * x2
        return torch.stack([dx1, dx2, dx3], dim=-1)


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

    A model-based classifier (odeint/Kalman) that knows the candidate fields identifies the
    coupling exactly: for these predictable (non-chaotic) systems the per-step prediction is
    sharp, so the multi-step likelihood-ratio between candidates grows ~linearly with the
    trajectory length. Set ``decay`` small (e.g. 0.1) for sustained, persistently-informative
    oscillations; larger ``decay`` makes the discriminative signal live in an early transient.
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
