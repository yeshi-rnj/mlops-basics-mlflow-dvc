"""Export an MLflow runs summary and draw the comparison plots used in the README."""
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from tracking import SUMMARY_CSV, load_runs, short_params

SHOW = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main():
    runs = load_runs()
    runs["short_params"] = runs.apply(short_params, axis=1)

    keep = ["run_id", "run_name"] + [c for c in runs.columns
                                      if c.startswith(("params.", "metrics."))]
    runs[keep].to_csv(SUMMARY_CSV, index=False)

    table = pd.DataFrame({
        "Run": runs["run_name"],
        "Model": runs["params.model_type"],
        "Params": runs["short_params"],
        **{m: runs[f"metrics.{m}"].round(4) for m in SHOW},
    })
    table.to_markdown(config.RESULTS_DIR / "runs_summary.md", index=False) \
        if _has_tabulate() else None
    best = runs.sort_values(["metrics.f1", "metrics.accuracy"], ascending=False).iloc[0]
    print(table.to_string(index=False))
    print(f"\nBest run: {best['run_name']}  (f1={best['metrics.f1']:.4f})")

    # 02 - grouped bars
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(runs))
    w = 0.26
    for i, m in enumerate(["accuracy", "f1", "roc_auc"]):
        bars = ax.bar(x + (i - 1) * w, runs[f"metrics.{m}"], w, label=m)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0008,
                    f"{b.get_height():.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(runs["run_name"], rotation=15, ha="right")
    lo = min(runs[f"metrics.{m}"].min() for m in ["accuracy", "f1", "roc_auc"])
    ax.set_ylim(max(0.0, lo - 0.03), 1.012)
    ax.set_ylabel("score (test set)")
    ax.set_title("MLflow runs - test metrics comparison", pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(config.IMG_DIR / "02_runs_comparison.png", dpi=150)
    plt.close(fig)

    # 03 - exported summary table
    fig, ax = plt.subplots(figsize=(12, 0.6 + 0.5 * len(table)))
    ax.axis("off")
    cell = table.astype(str).values.tolist()
    tb = ax.table(cellText=cell, colLabels=list(table.columns), loc="center",
                  cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.5)
    tb.auto_set_column_width(list(range(len(table.columns))))
    best_idx = list(runs["run_name"]).index(best["run_name"])
    for c in range(len(table.columns)):
        tb[0, c].set_facecolor("#dfe8f5")
        tb[best_idx + 1, c].set_facecolor("#e3f3e6")
    ax.set_title("Exported MLflow run summary (best run highlighted)", pad=8)
    fig.savefig(config.IMG_DIR / "03_runs_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 04 - ROC curves of all runs
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for name in runs["run_name"]:
        f = config.OUTPUT_DIR / name / "roc.json"
        if f.exists():
            d = json.loads(f.read_text())
            auc = runs.loc[runs.run_name == name, "metrics.roc_auc"].iloc[0]
            ax.plot(d["fpr"], d["tpr"], label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - all runs")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.IMG_DIR / "04_roc_curves.png", dpi=150)
    plt.close(fig)

    # 05 - confusion matrix of the best run
    cm = config.OUTPUT_DIR / best["run_name"] / "confusion_matrix.png"
    if cm.exists():
        shutil.copy(cm, config.IMG_DIR / "05_confusion_matrix_best.png")

    print(f"Wrote {SUMMARY_CSV} and images in {config.IMG_DIR}")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
