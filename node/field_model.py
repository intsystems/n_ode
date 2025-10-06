from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class VectorFieldMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int = 5,
            activation: type[nn.Module] = nn.Tanh,
            activation_params: dict = {}
        ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(**activation_params)
        )

        for i in range(num_layers):
            if i % 2 == 0:
                # explictly name each batch norm
                # for convinience in futher pyro extension
                # self.layers.add_module(f"batch_norm_{i}", nn.BatchNorm1d(hidden_dim))
                self.layers.append(nn.Dropout1d(p=0.05))

            self.layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                activation(**activation_params)]
            )
        
        self.layers.append(nn.Linear(hidden_dim, input_dim))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class VectorFieldLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

        # inital point is zero
        self.weight = nn.Parameter(
            torch.zeros_like(self.weight, device=self.weight.device)
        )
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
