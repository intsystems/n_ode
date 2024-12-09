from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class VectorFieldMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 5):
        super().__init__()

        activation = nn.LeakyReLU(0.5)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            deepcopy(activation)
        )

        for _ in range(num_layers):
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), deepcopy(activation)])
        
        self.layers.append(nn.Linear(hidden_dim, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class VectorFieldLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

        # inital point is zero
        self.weight = nn.Parameter(
            torch.zeros_like(self.weight, device=self.weight.device)
        )
