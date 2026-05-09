import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
from itertools import chain
from toolz import pipe
from toolz.curried import map as map_c

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, ConcatDataset
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
# from torchmetrics.regression import ...

from experiment.motion_sense.utils.dataset import TrajectoryDataset
from experiment.motion_sense.utils.field import FieldLitModule

import plotly.graph_objects as go
import rich
from rich.progress import track
import mlflow



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subj", type=int)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/motion_sense/config.yaml")
    TRAJ_COMPONENT_TO_VIZ = 0

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("motion_sense")
    mlflow.start_run(
        run_name="subj_classifiy",
        tags={"subj": args.subj}
    )

    act_fields_adapters = {}
    test_trajectories = {}
    for act_dir in Path("fields").iterdir():
        act = act_dir.stem
        subj_field_path = act_dir / f"{args.subj}/best.ckpt"
        if subj_field_path.exists():
            act_fields_adapters[act] = FieldLitModule.load_from_checkpoint(
                subj_field_path, weights_only=False
            )

            test_dataset = TrajectoryDataset(
                config.data_dir, config.data_types, act,
                config.activity_codes[act][-1], args.subj
            )
            state_dim = test_dataset[0].shape[0] // 2
            test_trajectories[act] = torch.stack(
                [test_dataset[i][:state_dim] for i in range(len(test_dataset))]
            )
    
    # perform classification
    cls_results = []
    for target_act, test_traj in track(list(test_trajectories.items()), "Target act"):
        fig = go.Figure()
        t_mesh = np.arange(test_traj.shape[0]) * config.dt
        fig.add_trace(go.Scatter(
            x=t_mesh, 
            y=test_traj[:, TRAJ_COMPONENT_TO_VIZ],
            mode='lines',
            name="traj"
        ))

        for act, field_adapter in track(list(act_fields_adapters.items()), "Pred act", transient=True):
            field_adapter.traj_smooth_loss.reset()
            field_adapter.traj_smooth_loss.update(test_traj)
            loss, _, traj_smooth = field_adapter.traj_smooth_loss.compute()
            cls_results.append((target_act, act, loss.item()))

            fig.add_trace(go.Scatter(
                x=t_mesh, 
                y=traj_smooth[:, TRAJ_COMPONENT_TO_VIZ],
                mode='lines',
                name=f"{act}_smooth"
            ))
        
        state_name = field_adapter.state_names[TRAJ_COMPONENT_TO_VIZ]
        fig.update_layout(
            xaxis_title="t", yaxis_title=state_name
        )
        mlflow.log_figure(fig, f"{target_act}.html")

    cls_results = pd.DataFrame(cls_results, columns=["act_true", "act", "loss"])
    argmin_indx = cls_results.groupby("act_true")["loss"].idxmin()
    accuracy = (cls_results.loc[argmin_indx]["act"] == cls_results.loc[argmin_indx]["act_true"]).mean()
    mlflow.log_metric("accuracy", accuracy)

    save_dir = Path(os.path.join(config.resutls_dir, str(args.subj)))
    cls_results[argmin_indx].to_csv(save_dir / "cls.csv", index=False)

    print(accuracy)
