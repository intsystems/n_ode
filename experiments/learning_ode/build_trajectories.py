from pathlib import Path
from shutil import rmtree
import yaml

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import scipy.linalg as linalg

from src.data_loading import creat_time_series, set_data_types


def main():
    # load config files for data and pipeline
    with open("config.yaml", "r"), open("../../data/dataset_params.yaml") as f1, f2:
        config = yaml.full_load(f1)
        data_params = yaml.full_load(f2)
    
    # make dir for trajectories
    traj_dir = Path("trajectories/")
    if traj_dir.exists():
        rmtree(traj_dir)
    traj_dir.mkdir()
    
    for activity, act_codes in data_params["activity_codes"]:
        data_types_list = set_data_types([config["data_type"]])
        # get labeled magnitudes of chosen signal
        series_df = creat_time_series(
            "../../data",
            data_types_list,
            [activity],
            act_codes
        )
    
        # Last activity-code is left for test, others - for train
    
        def make_trajectories_dataset(is_train: bool) -> Dataset:
            trajectories = []
            # length of the trajectories
            durations = []
    
            if is_train:
                cur_act_codes = act_codes[:-1]
            else:
                cur_act_codes = [act_codes[-1]]
    
            for act_code in cur_act_codes:
                for i in range(data_params["num_participants"]):
                    # get time series for current participant and activity code
                    series = series_df.loc[
                        (series_df["id"] == i) & (series_df["trial"] == act_code),
                        [config["data_type"]]
                    ].values
    
                    # build trajectory matrix of the series
                    traj_matrix = linalg.hankel(
                        series[:config["trajectory_dim"]], series[config["trajectory_dim"] - 1:]
                    )
                    # transpose matrix so time axis = 0
                    trajectories.append(torch.from_numpy(traj_matrix.T))
    
                    durations.append(series.shape[0])
    
            # pad trajectories so they have equal time length
            # also add batch dimension
            max_duration = max(durations)
            for i, trajectory in enumerate(trajectories):
                cur_duration = trajectory.shape[0]
                trajectories[i] = torch.concat([
                    trajectory, torch.zeros((max_duration - cur_duration, config["trajectory_dim"]))
                ])
                trajectories[i].unsqueeze_(0)
    
            # make torch Dataset
            trajectories = torch.concat(trajectories)
            durations = torch.FloatTensor(durations)
            return TensorDataset((trajectories, durations))
    
    
        # save train and test datasets on disk
        torch.save(make_trajectories_dataset(is_train=True), traj_dir / f"{activity}_train.pt")
        torch.save(make_trajectories_dataset(is_train=False), traj_dir / f"{activity}_test.pt")


if __name__ == "__main__":
    main()



    
    
    









