import torch
from torch.utils.data import Dataset

from experiment.motion_sense.utils.raw_data_loading import create_time_series, set_data_types


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir: str, data_types: list[str], act, act_code, subj):
        dt_columns = set_data_types(data_types)
        act_labels = [act]
        trial_codes = [[act_code]]
        ts_df = create_time_series(
            data_dir, dt_columns, act_labels, trial_codes, mode="raw"
        )
        ts_df = ts_df[ts_df["id"].isin([subj])]
        cols = [f"rotationRate.{axis}" for axis in "xyz"] + \
            [f"userAcceleration.{axis}" for axis in "xyz"]
        ts_df = ts_df[cols]

        start_point = torch.from_numpy(ts_df.to_numpy())[:-1]
        next_point = torch.from_numpy(ts_df.shift(-1).to_numpy())[:-1]
        self.start_next_point = torch.concat((start_point, next_point), dim=1)
        # trajectory dim
        self.d = len(cols)

    def __getitem__(self, index):
        return self.start_next_point[index]

    def __len__(self):
        return self.start_next_point.shape[0]
