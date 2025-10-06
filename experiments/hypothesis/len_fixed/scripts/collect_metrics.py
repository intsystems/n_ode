""" Script to launch classification with gaussian bayes.
    Editable.
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import pandas as pd
import plotly.express as px

import wandb
from wandb.util import generate_id

from node.metrics import compute_agragate_metrics, make_recall_hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wandb_config", type=Path)
    parser.add_argument("classify_tables_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    wandb_config: DictConfig = OmegaConf.load(args.wandb_config)

    run = wandb.init(
        name="results-" + generate_id(),
        **dict(wandb_config)
    )

    lh_df = []
    for classify_table_path in args.classify_tables_dir.glob("*"):
        cur_table = pd.read_csv(classify_table_path)
        cur_subj_id = cur_table.loc[0, "subj"]
        lh_df.append(cur_table)
        run.use_artifact(f"node/subj_{cur_subj_id}_classify_table:latest")
    lh_df = pd.concat(lh_df, ignore_index=True)

    lh_df["pred"] = lh_df.drop(columns=["test_act", "traj_num", "subj"]).idxmin(axis=1)
    lh_df.rename(columns={"test_act": "true"}, inplace=True)

    agr_metrics = []
    recall_hists = []

    # compute per-subject metrics
    for subj_id, lh_df_subj in lh_df.groupby("subj"):
        agr_metrics_subj = compute_agragate_metrics(lh_df_subj["true"], lh_df_subj["pred"], "hype")
        agr_metrics_subj["subj"] = subj_id
        agr_metrics.append(agr_metrics_subj)
        recall_hists_subj = make_recall_hist(lh_df_subj["true"], lh_df_subj["pred"], "hype")
        recall_hists_subj["subj"] = subj_id
        recall_hists.append(recall_hists_subj)

    # compute global metrics
    agr_metrics_all = compute_agragate_metrics(lh_df["true"], lh_df["pred"], "hype")
    agr_metrics_all["subj"] = "all"
    agr_metrics.append(agr_metrics_all)
    recall_hists_all = make_recall_hist(lh_df["true"], lh_df["pred"], "hype")
    recall_hists_all["subj"] = "all"
    recall_hists.append(recall_hists_all)

    agr_metrics = pd.concat(agr_metrics, ignore_index=True)
    agr_metrics["subj"] = agr_metrics["subj"].astype(str)
    agr_metrics.to_csv(args.results_dir / "agr_metrics.csv", index=False)
    run.log({"agr_metrics_table": wandb.Table(dataframe=agr_metrics)})

    agr_metrics.drop(columns=["method"], inplace=True)
    agr_metrics = pd.melt(agr_metrics, id_vars=["subj"], var_name="metric")
    metrics_bar = px.bar(agr_metrics, x="metric", y="value", color="subj", barmode="group")
    metrics_bar.write_html(args.results_dir / "agr_metrics.html")
    run.log({"agr_metrics_hist": wandb.Plotly(metrics_bar)})

    recall_hists = pd.concat(recall_hists, ignore_index=True)
    fig = px.bar(recall_hists, x="act", y="recall", color="subj", barmode="group")
    fig.write_html(args.results_dir / "recall_hist.html")
    run.log({"recall_hists": wandb.Plotly(fig)})

    run.alert("Hypothesis experiment", "Finished")
