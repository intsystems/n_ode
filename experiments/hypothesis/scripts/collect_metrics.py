""" Script to launch classification with gaussian bayes.
    Editable.
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf

import pandas as pd
import plotly.express as px

import wandb
from wandb.util import generate_id

from node.metrics import compute_agragate_metrics, make_recall_hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("classify_table", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("config_wandb_path", type=Path)
    args = parser.parse_args()

    wandb_config = OmegaConf.load(args.config_wandb_path)

    run = wandb.init(
        name="results-" + generate_id(),
        **dict(wandb_config)
    )

    lh_df = pd.read_csv(args.classify_table)
    run.use_artifact("classify_table:latest")

    lh_df["pred"] = lh_df.drop(columns=["test_act", "traj_num"]).idxmin(axis=1)
    lh_df.rename(columns={"test_act": "true"}, inplace=True)

    agr_metrics = compute_agragate_metrics(lh_df["true"], lh_df["pred"], "hype")
    recall_hists = make_recall_hist(lh_df["true"], lh_df["pred"], "hype")

    agr_metrics.to_csv(args.results_dir / "agr_metrics.csv", index=False)
    run.log({"agr_metrics": wandb.Table(dataframe=agr_metrics)})

    print(recall_hists)
    fig = px.bar(recall_hists, x="act", y="recall", color="method", barmode="group")
    fig.write_html(args.results_dir / "recall_hist.html")
    run.log({"recall_hists": wandb.Plotly(fig)})

    run.alert("Hypothesis experiment", "Finished")
