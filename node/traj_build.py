from pathlib import Path
from shutil import rmtree
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import scipy.linalg as linalg

from .raw_data_loading import creat_time_series, set_data_types


def slice_trajectory(traj_matrix: np.ndarray, max_len: int) -> dict:
    """ legacy function for slicing phase trajectory on patches with max_len length
    """
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


def normalize_trajectories(traj_dataset: TensorDataset) -> Dataset:
    trajectories = traj_dataset.tensors[0]

    mean = torch.mean(trajectories, dim=(0, 1))
    std = torch.std(trajectories, dim=(0, 1))
    normed_trajectories = (trajectories - mean) / std

    return TensorDataset(normed_trajectories, traj_dataset.tensors[1])


def make_activity_dataset(activity: str, traj_dir: Path) -> Dataset:
    act_files = list(traj_dir.glob(f"{activity}*"))
    traj_list = list(map(
        torch.load,
        act_files
    ))
    max_traj_len = max(
        list(map(
            lambda x: x.shape[0],
            traj_list
        ))
    )

    durations = torch.empty((len(traj_list), ), dtype=torch.int32)

    # pad trajectories to equal len
    for i, trajectory in enumerate(traj_list):
        cur_duration = trajectory.shape[0]
        durations[i] = cur_duration

        pad_len = max_traj_len - cur_duration
        if pad_len > 0:
            traj_list[i] = torch.concat(
                [traj_list[i], torch.zeros((pad_len, traj_list[i].shape[1]))]
            )
    # merge trajectories
    trajectories = torch.stack(traj_list)

    return TensorDataset(trajectories, durations)


def make_trajectories(config: dict, data_path: Path, traj_dir: Path = "trajectories/"):
    """Creates trajectories as torch.Tensors for each activity, activity code, participant.

    Args:
        config (dict): _description_
        data_path (Path): _description_
        traj_dir (Path, optional): dir to save trajectories. Defaults to "trajectories/".
    """
    print("Building phase trajectories...")

    # load config files for data
    with open(data_path / "dataset_params.yaml") as f2:
        data_params = yaml.full_load(f2)

    # make dir for trajectories
    if traj_dir.exists():
        rmtree(traj_dir)
    traj_dir.mkdir()

    for activity, act_codes in data_params["activity_codes"].items():
        # get labeled magnitudes of chosen signal
        series_df = creat_time_series(
            str(data_path),
            set_data_types([config["data_type"]]),
            [activity],
            [act_codes]
        )

        for act_code in act_codes:
            for partic_id in range(data_params["num_participants"]):
                print(f"Activity: {activity}; Act_code: {act_code}; Participant: {partic_id}")

                # get time series for current participant and activity code
                series = series_df.loc[
                    (series_df["id"] == partic_id) & (
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
                # transfer in torch
                trajectory = torch.from_numpy(traj_matrix).float()

                # save dataset on disk
                torch.save(
                    trajectory,
                    traj_dir / f"{activity}_{act_code}_{partic_id}.pt"
                )

# # load config files for data and pipeline
# with open("config.yaml", "r") as f1:
#     config = yaml.full_load(f1)

# make_trajectories(
#     config,
#     Path("../data"),
#     Path("../data/trajectories")
# )
