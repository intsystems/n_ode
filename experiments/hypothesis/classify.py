""" launches neural ode train processes for specific activity label
"""
from pathlib import Path
import wandb.sdk
import wandb.wandb_run
import yaml

import wandb

import numpy as np
from torchdyn.core import NeuralODE

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from node.field_model import VectorFieldMLP
