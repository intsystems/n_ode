import torch
from torch import nn
from torch import optim


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
