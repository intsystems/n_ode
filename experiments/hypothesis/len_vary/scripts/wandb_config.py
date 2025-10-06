""" Creates global wandb config for the whole pipeline. Editable.
"""
import wandb
import yaml
from snakemake.script import snakemake
import wandb.util

if __name__ == "__main__":
    wandb_config = {
        "project": "node",
        "group": "hypothesis-vary-"+  wandb.util.generate_id(length=4),
        # "mode": "disabled"
    }

    with open(snakemake.output[0], "w") as f:
        yaml.dump(wandb_config, f)
