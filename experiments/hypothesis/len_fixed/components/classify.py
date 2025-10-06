from pipe import select

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from rich.console import Console
from rich.progress import track

from node.data_modules import ActivityDataModule
from node.field_module import get_trajectory_mask
from .field_module import LitNodeHype

console = Console()

# launches models on each activity and stores liklyhood of each trajectory
@torch.no_grad
def build_lh_table(
    test_datasets: dict[str, ActivityDataModule],
    models: dict[str, LitNodeHype]
) -> pd.DataFrame:
    act_lh = []

    for test_act, test_dataset in test_datasets.items():
        test_act_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # lh for each model for current activity data
        models_lh = {model_act: [] for model_act in models.keys()}

        cur_traj_num = -1
        for traj, duration, subj, traj_num in track(test_act_loader, f"Classifying {test_act}..."):
            # remove batch dim
            traj_num: int = traj_num.item()
            if cur_traj_num != traj_num:
                any(models_lh.values() | select(lambda l: l.append(0.0)))
                cur_traj_num = traj_num

            for model_act, model in models.items():
                z_0 = traj[:, 0, :]
                pred = model(z_0, num_steps=traj.shape[1])
                mask = get_trajectory_mask(duration, traj)
                lk_func = list(model.loss_funcs.values())[0]
                # compute normed -1 * likelihood
                lh = lk_func(traj, pred * mask)
                models_lh[model_act][-1] += lh.item()

        models_lh = pd.DataFrame(models_lh)
        models_lh["test_act"] = test_act
        models_lh.reset_index(inplace=True, names="traj_num")

        act_lh.append(models_lh)

    act_lh = pd.concat(act_lh, axis=0, ignore_index=True)
    return act_lh
