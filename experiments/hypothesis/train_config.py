from pathlib import Path
import yaml
import wandb.sdk
import wandb.wandb_run
import wandb

import numpy as np
from torchdyn.core import NeuralODE

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from node.field_model import VectorFieldMLP

import warnings
warnings.simplefilter("ignore", FutureWarning)

def get_model() -> NeuralODE:
    """ Default vector field and node
    """
    # load config files for pipeline
    with open("config.yaml", "r") as f1:
        config = yaml.full_load(f1)

    vector_field = VectorFieldMLP(**config["model"])
    return NeuralODE(
        vector_field,
        **config["node"]
    )


def get_optimizer(ode_model: NeuralODE, act: str) -> optim.Optimizer:
    """ Default optimizer for each activities. It can be tuned for each activity
    """
    # load config files for pipeline
    with open("config.yaml", "r") as f1:
        config = yaml.full_load(f1)

    return optim.Adam(
        ode_model.parameters(),
        **config["optim"]
    )


@torch.no_grad
def vizualize_pred_traj(
    act: str,
    ode_model: NeuralODE,
    test_loader: DataLoader,
    run: wandb.sdk.wandb_run.Run
):
    device = ode_model.device

    # get first <=3 test trajectories
    traj, durations = next(iter(test_loader))
    num_traj = min(3, traj.shape[0])

    traj = traj[:num_traj].to(device)
    durations = durations[:num_traj].to(device)

    # get prediction
    traj_len = traj.shape[1]
    t_span = torch.arange(0, traj_len).to(device)
    _, traj_predict = ode_model(traj[:, 0, :], t_span)
    # move batch axis in front
    traj_predict = traj_predict.movedim(1, 0)

    for i in range(num_traj):
        # vizualize first two components of the trajectory vector
        true_traj = traj[i, :durations[i], :2].cpu().numpy()
        pred_traj = traj_predict[i, :durations[i], :2].cpu().numpy()

        run.log(
            {
                f"{act}_traj_{i}": wandb.plot.line_series(
                    [true_traj[:, 0], pred_traj[:, 0]],
                    [true_traj[:, 1], pred_traj[:, 1]],
                    keys=["true", "pred"],
                    title=f"Trajectories {i}"
                )
            }
        )


# walkaround to add new callbacks in training process
# e.g. to add "download model" callback in colab
additional_callbacks = {}


def get_callbacks(
    run: wandb.sdk.wandb_run.Run,
    act: str,
    test_loader: DataLoader,
    models_dir: Path
) -> dict:
    """ Default callbacks for training + importing additional callbacks
    """
    prev_loss = None
    prev_prev_loss = None

    def pre_epoch(ode_model: NeuralODE) -> bool:
        nonlocal prev_prev_loss
        nonlocal prev_loss

        if prev_loss is None or prev_prev_loss is None:
            return False
        else:
            return np.abs(prev_loss - prev_prev_loss) < 1e-3
        
    def train(ode_model: NeuralODE, results: dict):
        for label, val in results.items():
            run.log({f"Train/{act}_{label}": val})
        
        nonlocal prev_prev_loss
        nonlocal prev_loss
        prev_prev_loss = prev_loss
        prev_loss = results["mse"]

    def test(ode_model: NeuralODE, results: dict):
        for label, val in results.items():
            run.log({f"Test/{act}_{label}": val})

    def post_epoch(ode_model: NeuralODE):
        vizualize_pred_traj(act, ode_model, test_loader, run)

        # log model
        torch.save(ode_model, models_dir / f"{act}.pt")
        run.log_model(path=models_dir / f"{act}.pt", name=act)

    callbacks = {
        "pre_epoch": [pre_epoch],
        "train": [train],
        "test": [test],
        "post_epoch": [post_epoch]
    }

    # import additional callbacks
    for add_cb_key, add_cb_list in additional_callbacks.items():
        callbacks[add_cb_key].extend(add_cb_list, **locals())

    return callbacks
