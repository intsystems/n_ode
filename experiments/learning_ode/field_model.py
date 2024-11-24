import torch
from torch import nn
from torch.nn import functional as F


class VectorFieldMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        for i in range(2):
            if i % 2 == 0:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
                
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        
        self.layers.append(nn.Linear(hidden_dim, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    

class VectorFieldLinear(nn.Linear):
    pass
