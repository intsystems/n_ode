from pathlib import Path
from shutil import rmtree
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import scipy.linalg as linalg

from src.data_loading import creat_time_series, set_data_types


def slice_trajectory(traj_matrix: np.ndarray, max_len: int) -> dict:
    num_slices = traj_matrix.shape[0] // max_len
    traj_residue_len = traj_matrix.shape[0] % max_len

    if num_slices == 0:
        residue_dict = {
            "residue": {
                "traj": torch.from_numpy(np.expand_dims(traj_matrix, axis=0)),
                "duration": torch.from_numpy(
                    np.array([traj_residue_len, ], dtype=np.int32)
                )
            }
        }
        return residue_dict
    
    maxlen_traj = traj_matrix[:traj_matrix.shape[0] - traj_residue_len].reshape(num_slices, max_len, -1)
    out_dict = {
        "full": {
            "traj": torch.from_numpy(maxlen_traj),
            "duration": torch.from_numpy(
                np.array([max_len, ], dtype=np.int32).repeat(num_slices)
            )
        }
    }
    if traj_residue_len != 0:
        residue_traj = np.expand_dims(
            traj_matrix[-traj_residue_len:],
            axis=0
        )
        out_dict.update({
            "residue": {
                "traj": torch.from_numpy(residue_traj),
                "duration": torch.from_numpy(
                    np.array([traj_residue_len, ], dtype=np.int32)
                )
            }
        })

    return out_dict


def main():
    print("Building phase trajectories.")

    # load config files for data and pipeline
    with open("config.yaml", "r") as f1:
        config = yaml.full_load(f1)
    with open("../../data/dataset_params.yaml") as f2:
        data_params = yaml.full_load(f2)

    # make dir for trajectories
    traj_dir = Path("trajectories/")
    if traj_dir.exists():
        rmtree(traj_dir)
    traj_dir.mkdir()

    for activity, act_codes in data_params["activity_codes"].items():
        # get labeled magnitudes of chosen signal
        series_df = creat_time_series(
            "../../data",
            set_data_types([config["data_type"]]),
            [activity],
            [act_codes]
        )

        # Last activity-code is left for test, others - for train

        def make_trajectories_dataset(is_train: bool) -> Dataset:
            trajectories = []
            # length of the trajectories
            durations = []
            # unified traj. matrix for computing mean and std
            uni_traj_matrix = []

            if is_train:
                cur_act_codes = act_codes[:-1]
            else:
                cur_act_codes = [act_codes[-1]]

            for act_code in cur_act_codes:
                for i in range(data_params["num_participants"]):
                    # get time series for current participant and activity code
                    series = series_df.loc[
                        (series_df["id"] == i) & (
                            series_df["trial"] == act_code),
                        [config["data_type"]]
                    ].values

                    # build trajectory matrix of the series
                    traj_matrix = linalg.hankel(
                        series[:config["trajectory_dim"]
                               ], series[config["trajectory_dim"] - 1:]
                    ).astype(np.float32)
                    # transpose matrix so time axis = 0
                    traj_matrix = traj_matrix.T
                    uni_traj_matrix.append(traj_matrix)

                    for traj_info in slice_trajectory(traj_matrix, config["trajectory_len"]).values():
                        trajectories.append(traj_info["traj"])
                        durations.append(traj_info["duration"])

            # compute mean and std of trajectory vectors
            uni_traj_matrix = np.concat(uni_traj_matrix, axis=0)
            traj_mean = torch.from_numpy(
                uni_traj_matrix.mean(axis=0)
            )
            traj_std = torch.from_numpy(
                uni_traj_matrix.std(axis=0, ddof=1)
            )
            # save these statistics
            if is_train:
                file_suffix = "train.pt"
            else:
                file_suffix = "test.pt"
            torch.save(traj_mean, traj_dir / f"{activity}_mean_{file_suffix}")
            torch.save(traj_std, traj_dir / f"{activity}_std_{file_suffix}")

            # normalize trajectores and pad them so they have equal time length
            for i, trajectory in enumerate(trajectories):
                trajectories[i] = (trajectories[i] - traj_mean) / traj_std

                cur_duration = trajectory.shape[1]
                if cur_duration < config["trajectory_len"]:
                    trajectories[i] = torch.concat(
                        [trajectory, torch.zeros(
                            (1, config["trajectory_len"] - cur_duration, config["trajectory_dim"]), dtype=torch.float32)],
                        dim=1
                    )

            # make torch Datasets
            trajectories = (torch.concat(trajectories))
            durations = torch.concat(durations)
            return TensorDataset(trajectories, durations)

        # save train and test datasets on disk
        torch.save(make_trajectories_dataset(is_train=True),
                   traj_dir / f"{activity}_train.pt")
        torch.save(make_trajectories_dataset(is_train=False),
                   traj_dir / f"{activity}_test.pt")


if __name__ == "__main__":
    main()
