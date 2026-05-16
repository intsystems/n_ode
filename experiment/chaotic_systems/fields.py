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
