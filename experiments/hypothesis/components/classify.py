from pipe import select

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from node.data_modules import ActivityDataModule
from node.field_module import get_trajectory_mask
from .field_module import LitNodeHype, compute_lh


# launches models on each activity and stores liklyhood of each trajectory
@torch.no_grad
def build_lh_table(
    datamodules: dict[str, ActivityDataModule],      # for validation data; activity -> datamodule
    models: dict[str, LitNodeHype]
) -> pd.DataFrame:
    act_lh = []

    for test_act in datamodules.keys():
        test_act_loader = DataLoader(datamodules[test_act].val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

        # lh for each model for current activity data
        models_lh = {model_act: [] for model_act in models.keys()}

        cur_traj_num = -1
        for traj, duration, subj, traj_num in test_act_loader:
            # remove batch dim
            traj_num: int = traj_num.item()
            if cur_traj_num != traj_num:
                any(models_lh.values() | select(lambda l: l.append(0.0)))

            for model_act, model in models.items():
                z_0 = traj[:, 0, :]
                pred = model(z_0, num_steps=traj.shape[1])
                mask = get_trajectory_mask(duration, traj)
                lh = compute_lh(traj, pred, mask)
                models_lh[model_act][-1] += lh.item()

        models_lh = pd.DataFrame(models_lh)
        models_lh["test_act"] = test_act
        models_lh.reset_index(inplace=True, names="traj_num")

        act_lh.append(models_lh)

    act_lh = pd.concat(act_lh, axis=0, ignore_index=True)
    return act_lh
