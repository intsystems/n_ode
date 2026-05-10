import argparse
import os
from pathlib import Path
from itertools import chain
from toolz import pipe
from toolz.curried import map as map_c
from omegaconf import OmegaConf

import torch
import numpy as np
import pandas as pd

from experiment.motion_sense.utils.raw_data_loading import create_time_series, set_data_types

WINDOW_SIZE = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("act", type=str)
    parser.add_argument("subj", type=int)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/motion_sense/config.yaml")

    dt_columns = set_data_types(config.data_types)
    act_labels = [args.act]
    ts_df = create_time_series(
        config.data_dir, dt_columns, [args.act],
        [config.activity_codes[args.act]], mode="raw"
    )
    ts_df = ts_df[ts_df["id"].isin([args.subj])]

    cols = config.state_names
    noise = []
    for _, group in ts_df.groupby("trial"):
        group = group[cols]
        noise.append(
            (group - group.rolling(WINDOW_SIZE).mean()).dropna()
        )
    noise = pd.concat(noise)
    noise_sigma = noise.std(axis=0)

    save_dir = Path(os.path.join(config.results_dir, str(args.subj), args.act))
    save_dir.mkdir(parents=True, exist_ok=True)
    noise_sigma.to_csv(save_dir / "traj_std.csv")
