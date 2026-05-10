from itertools import chain
from toolz import pipe
from toolz.curried import map as map_c

import torch
from torch.utils.data import Dataset

from experiment.motion_sense.utils.raw_data_loading import create_time_series, set_data_types


class TrajectoryDataset(Dataset):
    def __init__(
        self, data_dir: str, data_types: list[str], state_names: list[str],
        act, act_code, subj, window_size: int = 16
    ):
        dt_columns = set_data_types(data_types)
        act_labels = [act]
        trial_codes = [[act_code]]
        ts_df = create_time_series(
            data_dir, dt_columns, act_labels, trial_codes, mode="raw"
        )
        ts_df = ts_df[ts_df["id"].isin([subj])]
        ts_df = ts_df[state_names]

        self.traj = torch.from_numpy(ts_df.to_numpy())
        self.window_size = window_size
        self.d = len(state_names)

    def __getitem__(self, index):
        return self.traj[index : index + self.window_size]

    def __len__(self):
        return self.traj.shape[0] - self.window_size + 1
