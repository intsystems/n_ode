""" Script to launch dataset building.
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from pipe import select

import torch
from torch.utils.data import TensorDataset

from node.raw_data_loading import create_time_series, set_data_types
from node.traj_build import takens_traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subj_id", type=int)
    parser.add_argument("act", type=str)
    parser.add_argument("data_config_path", type=Path)
    parser.add_argument("raw_data_dir", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    # load config file
    data_config: DictConfig = OmegaConf.load(args.data_config_path)

    torch.manual_seed(data_config.seed)

    def build_trajs():
        # load config file for data
        data_params = OmegaConf.load(args.raw_data_dir / "dataset_params.yaml")

        # get labeled magnitudes of chosen signal
        series_df = create_time_series(
            str(args.raw_data_dir),
            set_data_types([data_config.data.data_type]),
            [args.act],
            [data_params.activity_codes[args.act]]
        )

        data = []
        # initial trajectory number for slices
        traj_num = 0
        for act_code in data_params.activity_codes[args.act]:
            def build_sliced_takens_traj():
                # get time series for current participant and activity code
                series = series_df.loc[
                    (series_df["id"] == args.subj_id) & (
                        series_df["trial"] == act_code),
                    [data_config.data.data_type]
                ].values

                cur_data = takens_traj(series, data_config.data.dim, data_config.data.max_len)
                # transform to torch
                cur_data = (
                    torch.tensor(cur_data[0], dtype=torch.float32),
                    torch.tensor(cur_data[1], dtype=torch.int)
                )
                # add subject info to tensor dataset
                num_trajs = cur_data[0].shape[0]
                cur_data = list(cur_data) + [torch.tensor([args.subj_id] * num_trajs)]
                return cur_data

            cur_data = build_sliced_takens_traj()
            num_trajs = cur_data[0].shape[0]
            # add trajectory num info to tensor dataset
            cur_data += [torch.tensor([traj_num] * num_trajs)]

            names = ["traj", "dur", "subj_id", "traj_num"]
            data.append({
                name: v for (name, v) in zip(names, cur_data)
            })

            traj_num += 1

        return data
    
    sliced_trajs_list = build_trajs()
    # concat trajectories
    sliced_trajs = {}
    tensor_names = ["traj", "dur", "subj_id", "traj_num"]
    for name in tensor_names:
        sliced_trajs[name] = torch.concat(list(
            sliced_trajs_list | select(lambda t: t[name])
        ))
    del sliced_trajs_list

    # normalize trajectories
    sliced_trajs["traj"] = \
        (sliced_trajs["traj"] - sliced_trajs["traj"].mean(dim=(0, 1))) \
            / sliced_trajs["traj"].std(dim=(0, 1))

    dataset = TensorDataset(*list(sliced_trajs.values()))
    torch.save(dataset, args.output_file)
