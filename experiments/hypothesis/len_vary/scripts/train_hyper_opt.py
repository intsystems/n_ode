""" Launches vector field hyperparameter optimiazation.
    Editable.
"""
import argparse
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from components.field_model import MyVectorField
from components.field_module import LitNodeHype

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def objective(
    trial: optuna.Trial,
    train_config: DictConfig,
    test_dataset: Path
) -> float:
    hidden_dim = trial.suggest_int("hidden_dim", 20, 80, step=10)
    num_layers = trial.suggest_int("num_layers", 2, 8)
    
    train_config.vf.hidden_dim = hidden_dim
    train_config.vf.num_layers = num_layers
    vf = MyVectorField(**dict(train_config.vf))
    loss_funcs = {
        "MSE": F.mse_loss,
        "MAE": F.l1_loss
    }
    lit_node = LitNodeHype(vf, loss_funcs, dict(train_config.optim), dict(train_config.odeint))

    # load train/test datasets
    train_dataset: TensorDataset = torch.load(args.train_dataset, weights_only=False)
    train_loader = DataLoader(train_dataset, train_config.batch_size, shuffle=True)
    test_dataset: TensorDataset = torch.load(args.test_dataset, weights_only=False)
    test_loader = DataLoader(test_dataset, train_config.batch_size, shuffle=False)

    prune_callback = PyTorchLightningPruningCallback(trial, monitor="Val/MSE")
    earlystopping_callback = EarlyStopping(**dict(train_config.early_stop))

    trainer = L.Trainer(
        logger=False,
        callbacks=[prune_callback, earlystopping_callback],
        accelerator="cpu",
        devices=4,
        max_epochs=20,
        enable_checkpointing=False,
        strategy="ddp_spawn"
    )

    trainer.fit(lit_node, train_loader, test_loader)

    prune_callback.check_pruned()

    return trainer.callback_metrics["Val/MSE"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("act", type=str)
    parser.add_argument("train_config", type=Path)
    parser.add_argument("train_dataset", type=Path)
    parser.add_argument("test_dataset", type=Path)
    args = parser.parse_args()

    # load config files
    train_config: DictConfig = OmegaConf.load(args.train_config)

    objective_f = partial(objective, train_config=train_config, test_dataset=args.test_dataset)
    study = optuna.create_study(
        study_name=f"hyper_optim_{args.act}",
        storage="postgresql://myuser@localhost:5432/mydatabase",
        load_if_exists=True
    )
    study.optimize(objective_f, n_trials=10)

    print("Optimal hyperparams")
    for key, value in study.best_params.items():
        print("{}: {}".format(key, value))
