"""Train one model and log params, metrics and the model artifact to MLflow.

Example:
    python src/train.py --model rf --run-name rf_n300 \
        --param n_estimators=300 --param max_depth=None
"""
import argparse
import ast
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

import config
from models import fit_and_evaluate


def run_training(model_type, params, run_name, seed=config.SEED):
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    model, metrics, preds, train_time, params = fit_and_evaluate(model_type, params, seed)

    run_dir = config.OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix + ROC points (kept locally and logged as artifacts)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        preds["y_true"], preds["y_pred"], display_labels=config.CLASS_NAMES,
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {run_name}")
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_score"])
    (run_dir / "roc.json").write_text(
        json.dumps({"fpr": fpr.tolist(), "tpr": tpr.tolist()}))

    # Trained model file -> versioned by DVC (models/ is tracked with `dvc add`)
    model_path = config.MODEL_DIR / f"{run_name}.pkl"
    joblib.dump(model, model_path)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("seed", seed)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_artifacts(str(run_dir), artifact_path="plots")
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    print(f"[{run_name}] run_id={run_id}  acc={metrics['accuracy']:.4f}  "
          f"f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}")
    return run_id, metrics


def _parse_param(text):
    key, value = text.split("=", 1)
    try:
        return key, ast.literal_eval(value)      # 0.1, 300, None, ...
    except (ValueError, SyntaxError):
        return key, value


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["logreg", "rf", "gb"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--param", action="append", default=[],
                    help="hyperparameter as key=value (repeatable)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    run_training(args.model, dict(map(_parse_param, args.param)),
                 args.run_name, args.seed)
