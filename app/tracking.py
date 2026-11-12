"""Read logged runs back from MLflow (falls back to the exported CSV)."""
import ast

import pandas as pd

import config

SUMMARY_CSV = config.RESULTS_DIR / "runs_summary.csv"
HIDDEN_PARAMS = {"model_type", "seed"}


def load_runs():
    """One row per run_name (latest wins). Columns follow MLflow's naming:
    run_id, run_name, params.<k>, metrics.<k>."""
    df = pd.DataFrame()
    try:
        import mlflow
        mlflow.set_tracking_uri(config.TRACKING_URI)
        df = mlflow.search_runs(experiment_names=[config.EXPERIMENT_NAME])
    except Exception as exc:                                   # noqa: BLE001
        print(f"(MLflow store not readable: {exc})")

    if df is not None and len(df):
        df = df.rename(columns={"tags.mlflow.runName": "run_name"})
        df = df.sort_values("start_time").drop_duplicates("run_name", keep="last")
        return df.reset_index(drop=True)

    if SUMMARY_CSV.exists():
        print(f"No MLflow runs found - using {SUMMARY_CSV.name}")
        return pd.read_csv(SUMMARY_CSV)
    raise SystemExit("No runs found. Run: python src/run_experiments.py")


def run_params(row):
    """Logged params of a run as typed Python values (MLflow stores strings)."""
    out = {}
    for col, val in row.items():
        if col.startswith("params.") and pd.notna(val):
            key = col[len("params."):]
            try:
                out[key] = ast.literal_eval(str(val))
            except (ValueError, SyntaxError):
                out[key] = str(val)
    return out


def short_params(row):
    p = run_params(row)
    return ", ".join(f"{k}={v}" for k, v in p.items() if k not in HIDDEN_PARAMS)
