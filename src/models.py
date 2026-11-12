"""Data loading, model construction and evaluation (no MLflow in here)."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

DEFAULT_PARAMS = {
    "logreg": {"C": 1.0},
    "rf": {"n_estimators": 100, "max_depth": None},
    "gb": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
}


def load_data():
    df = pd.read_csv(config.DATA_PATH)
    X, y = df.drop(columns="target"), df["target"]
    return train_test_split(X, y, test_size=config.TEST_SIZE,
                            random_state=config.SEED, stratify=y)


def build_model(model_type, params, seed):
    if model_type == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=params["C"], max_iter=5000, random_state=seed))
    if model_type == "rf":
        return RandomForestClassifier(n_estimators=params["n_estimators"],
                                      max_depth=params["max_depth"],
                                      random_state=seed, n_jobs=1)
    if model_type == "gb":
        return GradientBoostingClassifier(n_estimators=params["n_estimators"],
                                          learning_rate=params["learning_rate"],
                                          max_depth=params["max_depth"],
                                          random_state=seed)
    raise ValueError(f"Unknown model_type: {model_type}")


def fit_and_evaluate(model_type, params, seed=config.SEED):
    """Train one model and return (model, metrics, predictions, train_time, params)."""
    params = {**DEFAULT_PARAMS[model_type], **params}
    X_train, X_test, y_train, y_test = load_data()
    model = build_model(model_type, params, seed)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
    }
    metrics = {k: float(v) for k, v in metrics.items()}
    preds = {"y_true": np.asarray(y_test), "y_pred": y_pred, "y_score": y_score}
    return model, metrics, preds, train_time, params
