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
        print(f"Activity: {activity}")

        # get labeled magnitudes of chosen signal
        series_df = creat_time_series(
            "../../data",
            set_data_types([config["data_type"]]),
            [activity],
            [act_codes]
        )

        for act_code in act_codes:
            print(f"Act_code: {act_code}")

            for partic_id in range(data_params["num_participants"]):
                print(f"Participant: {partic_id}")

                def make_partic_trajectory() -> Dataset:
                    # get time series for current participant and activity code
                    series = series_df.loc[
                        (series_df["id"] == partic_id) & (
                            series_df["trial"] == act_code),
                        [config["data_type"]]
                    ].values

                    # containers for trajectory patches
                    trajectories = []
                    durations = []

                    # build trajectory matrix of the series
                    traj_matrix = linalg.hankel(
                        series[:config["trajectory_dim"]
                               ], series[config["trajectory_dim"] - 1:]
                    ).astype(np.float32)
                    # transpose matrix so time axis = 0
                    traj_matrix = traj_matrix.T

                    for traj_info in slice_trajectory(traj_matrix, config["trajectory_len"]).values():
                        trajectories.append(traj_info["traj"])
                        durations.append(traj_info["duration"])

                    # compute mean and std of trajectory vectors
                    traj_mean = traj_matrix.mean(axis=0)
                    traj_std = traj_matrix.std(axis=0)

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

                    print(f"Num patches: {trajectories.shape[0]}")

                    return TensorDataset(trajectories, durations)

                # save dataset on disk
                torch.save(
                    make_partic_trajectory(),
                    traj_dir / f"{activity}_{act_code}_{partic_id}.pt"
                )


if __name__ == "__main__":
    main()
