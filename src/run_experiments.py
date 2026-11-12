"""Run the 5 tracked experiments (different models / hyperparameters)."""
import config
import prepare_data
from train import run_training

EXPERIMENTS = [
    ("logreg_C0.1",    "logreg", {"C": 0.1}),
    ("logreg_C10",     "logreg", {"C": 10.0}),
    ("rf_n50_d3",      "rf",     {"n_estimators": 50,  "max_depth": 3}),
    ("rf_n300_dNone",  "rf",     {"n_estimators": 300, "max_depth": None}),
    ("gb_n200_lr0.05", "gb",     {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3}),
]

if __name__ == "__main__":
    if not config.DATA_PATH.exists():
        prepare_data.main()
    for name, model_type, params in EXPERIMENTS:
        run_training(model_type, params, name)
    print("\nDone. Start the UI with:  mlflow ui --backend-store-uri", config.TRACKING_URI)
