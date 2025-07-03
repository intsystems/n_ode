""" Script to launch classification with gaussian bayes.
    Editable.
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf

import pandas as pd
import plotly.express as px

from node.metrics import compute_agragate_metrics, make_recall_hist

import wandb
from wandb.util import generate_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_wandb_path", type=Path)
    parser.add_argument("classify_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    wandb_config = OmegaConf.load(args.config_wandb_path)

    run = wandb.init(
        name="results-" + generate_id(),
        **dict(wandb_config)
    )

    agr_metrics = []
    recall_hists = []
    for classify_res_file in Path(args.classify_dir).glob("*"):
        method = classify_res_file.stem
    
        run.use_artifact(f"cls_result_{method}:latest")
        classify_df = pd.read_csv(classify_res_file)
        agr_metrics.append(
            compute_agragate_metrics(classify_df["true"], classify_df["pred"], method)
        )
        recall_hists.append(
            make_recall_hist(classify_df["true"], classify_df["pred"], method)
        )

    agr_metrics = pd.concat(agr_metrics, ignore_index=True)
    agr_metrics.to_csv(args.results_dir / "agr_metrics.csv", index=False)
    run.log({"agr_metrics": wandb.Table(dataframe=agr_metrics)})

    recall_hists = pd.concat(recall_hists, ignore_index=True)
    print(recall_hists)
    fig = px.bar(recall_hists, x="act", y="recall", color="method", barmode="group")
    fig.write_html(args.results_dir / "recall_hist.html")
    run.log({"recall_hists": wandb.Plotly(fig)})

    run.alert("Model space experiment", "Finished")
