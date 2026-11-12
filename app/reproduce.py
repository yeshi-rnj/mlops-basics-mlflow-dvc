"""Reproduce a past MLflow run from its logged parameters (+ `dvc pull`).

    python src/reproduce.py                       # best run
    python src/reproduce.py --run-name rf_n50_d3  # a specific run
    python src/reproduce.py --dvc-pull            # run `dvc pull` first
"""
import argparse
import json
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from models import fit_and_evaluate
from tracking import load_runs, run_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name")
    ap.add_argument("--dvc-pull", action="store_true",
                    help="restore dataset/models from the DVC remote first")
    args = ap.parse_args()

    if args.dvc_pull:
        if shutil.which("dvc") is None:
            raise SystemExit("dvc not found - pip install -r requirements.txt")
        subprocess.run(["dvc", "pull"], check=True)
    if not config.DATA_PATH.exists():
        raise SystemExit(f"{config.DATA_PATH} is missing - run `dvc pull` first.")

    runs = load_runs()
    if args.run_name:
        row = runs[runs["run_name"] == args.run_name]
        if row.empty:
            raise SystemExit(f"Run '{args.run_name}' not found: {list(runs.run_name)}")
        row = row.iloc[0]
    else:
        row = runs.sort_values(["metrics.f1", "metrics.accuracy"], ascending=False).iloc[0]

    params = run_params(row)
    model_type, seed = params.pop("model_type"), int(params.pop("seed"))
    print(f"Reproducing run '{row['run_name']}'  model={model_type}  seed={seed}")
    print(f"  logged params: {params}")

    _, new, _, _, _ = fit_and_evaluate(model_type, params, seed)

    rows, ok = [], True
    for k in config.METRIC_KEYS:
        old = float(row[f"metrics.{k}"])
        match = bool(np.isclose(old, new[k], atol=1e-9))
        ok &= match
        rows.append({"metric": k, "logged": old, "reproduced": new[k], "match": match})
        print(f"  {k:15s} logged={old:.6f}  reproduced={new[k]:.6f}  "
              f"{'OK' if match else 'MISMATCH'}")

    (config.RESULTS_DIR / "reproducibility_check.json").write_text(
        json.dumps({"run_name": row["run_name"], "params": params, "seed": seed,
                    "all_match": ok, "metrics": rows}, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x, w = np.arange(len(rows)), 0.38
    ax.bar(x - w / 2, [r["logged"] for r in rows], w, label="logged in MLflow")
    ax.bar(x + w / 2, [r["reproduced"] for r in rows], w, label="re-trained (dvc pull + params)")
    ax.set_xticks(x)
    ax.set_xticklabels([r["metric"] for r in rows], rotation=15)
    ax.set_ylim(0.9, 1.01)
    ax.set_title(f"Reproducibility check - {row['run_name']}: "
                 f"{'all metrics match' if ok else 'MISMATCH'}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.IMG_DIR / "06_reproducibility_check.png", dpi=150)
    plt.close(fig)

    print("\nREPRODUCED SUCCESSFULLY" if ok else "\nMETRICS DIFFER")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
