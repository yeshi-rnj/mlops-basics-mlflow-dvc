"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "breast_cancer.csv"      # versioned with DVC
MODEL_DIR = ROOT / "models"                           # versioned with DVC
OUTPUT_DIR = ROOT / "outputs" / "runs"                # per-run plots / roc points
IMG_DIR = ROOT / "docs" / "images"                    # images used in the README
RESULTS_DIR = ROOT / "results"

TRACKING_URI = (ROOT / "mlruns").as_uri()             # local MLflow file store
EXPERIMENT_NAME = "breast-cancer-classification"

SEED = 42
TEST_SIZE = 0.2
CLASS_NAMES = ["malignant", "benign"]                 # sklearn target: 0 / 1

# Metrics that must match exactly when a run is reproduced
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "train_accuracy"]

for _d in (MODEL_DIR, OUTPUT_DIR, IMG_DIR, RESULTS_DIR, DATA_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)
