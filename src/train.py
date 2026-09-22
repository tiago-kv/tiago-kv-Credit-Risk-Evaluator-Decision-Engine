"""Model training, hyperparameter tuning, and benchmarking module for Credit Risk Evaluator.

This script trains Logistic Regression (Baseline Scorecard), Random Forest, LightGBM,
and XGBoost using Stratified K-Fold cross validation. It optimizes the champion XGBoost
architecture via Optuna, evaluates banking-standard credit metrics (ROC-AUC, KS Statistic,
F1-Score, Recall, Precision), and exports production artifacts.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

# Ensure root package importability
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.data_processing import (
    create_preprocessor,
    get_feature_names,
    prepare_data,
)

# Configure optuna & logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Kolmogorov-Smirnov (KS) statistic between default and non-default classes.

    The KS statistic measures the maximum vertical distance between the cumulative
    distribution functions of good and bad borrowers. Standard metric under Basel regulations.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted default probabilities [0, 1].

    Returns:
        float: KS statistic value between 0.0 and 1.0.
    """
    try:
        bads = y_prob[y_true == 1]
        goods = y_prob[y_true == 0]
        if len(bads) == 0 or len(goods) == 0:
            return 0.0
        ks_stat = float(ks_2samp(bads, goods).statistic)
        return round(ks_stat, 4)
    except Exception as exc:
        logger.warning("Error calculating KS statistic: %s", str(exc))
        return 0.0


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate comprehensive credit risk classification metrics.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted default probabilities.
        threshold: Decision threshold for class assignment (default 0.50).

    Returns:
        Dict[str, float]: Metric dictionary containing ROC-AUC, KS, F1, Precision, Recall, Accuracy.
    """
    y_pred = (y_prob >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    ks = calculate_ks_statistic(y_true, y_prob)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    return {
        "roc_auc": round(auc, 4),
        "ks_statistic": round(ks, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "accuracy": round(acc, 4),
    }


def get_candidate_models(scale_pos_weight: float = 3.0) -> Dict[str, Any]:
    """Define dictionary of candidate models with configurations addressing class imbalance.

    Args:
        scale_pos_weight: Ratio of negative to positive classes for gradient boosters.

    Returns:
        Dict[str, Any]: Model name mapping to instantiated estimators.
    """
    return {
        "Logistic Regression (Baseline)": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.5,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def cross_validate_models(
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute Stratified K-Fold cross validation across all candidate models.

    Args:
        X_train_proc: Preprocessed feature matrix for training.
        y_train: Training labels array.
        n_splits: Number of CV folds (default 5).

    Returns:
        Tuple containing summary benchmark dataframe and dictionary of fitted CV estimators.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos = float(neg_count / max(pos_count, 1))

    candidate_models = get_candidate_models(scale_pos_weight=scale_pos)
    cv_records: List[Dict[str, Any]] = []

    for model_name, model in candidate_models.items():
        logger.info("Starting %d-fold cross-validation for %s...", n_splits, model_name)
        fold_auc, fold_ks, fold_f1, fold_prec, fold_rec = [], [], [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_proc, y_train)):
            X_f_train, X_f_val = X_train_proc[train_idx], X_train_proc[val_idx]
            y_f_train, y_f_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_f_train, y_f_train)
            val_probs = model.predict_proba(X_f_val)[:, 1]

            metrics = evaluate_predictions(y_f_val, val_probs)
            fold_auc.append(metrics["roc_auc"])
            fold_ks.append(metrics["ks_statistic"])
            fold_f1.append(metrics["f1_score"])
            fold_prec.append(metrics["precision"])
            fold_rec.append(metrics["recall"])

        mean_auc = float(np.mean(fold_auc))
        std_auc = float(np.std(fold_auc))
        mean_ks = float(np.mean(fold_ks))
        mean_f1 = float(np.mean(fold_f1))
        mean_prec = float(np.mean(fold_prec))
        mean_rec = float(np.mean(fold_rec))

        logger.info(
            "%s -> ROC-AUC: %.4f (±%.4f) | KS: %.4f | F1: %.4f | Rec: %.4f | Prec: %.4f",
            model_name,
            mean_auc,
            std_auc,
            mean_ks,
            mean_f1,
            mean_rec,
            mean_prec,
        )

        cv_records.append({
            "Model": model_name,
            "ROC-AUC (Mean)": round(mean_auc, 4),
            "ROC-AUC (Std)": round(std_auc, 4),
            "KS Statistic": round(mean_ks, 4),
            "F1-Score": round(mean_f1, 4),
            "Precision": round(mean_prec, 4),
            "Recall": round(mean_rec, 4),
        })

    benchmark_df = pd.DataFrame(cv_records).sort_values(by="ROC-AUC (Mean)", ascending=False)
    return benchmark_df, candidate_models


def tune_xgboost_optuna(
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    n_trials: int = 15,
) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters with Optuna targeting ROC-AUC.

    Args:
        X_train_proc: Preprocessed training features.
        y_train: Training labels array.
        scale_pos_weight: Imbalance compensation factor.
        n_trials: Number of Bayesian optimization trials.

    Returns:
        Dict[str, Any]: Dictionary of best hyperparameters found.
    """
    logger.info("Initiating Optuna hyperparameter tuning for XGBoost (%d trials)...", n_trials)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 350, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        auc_scores = []
        for train_idx, val_idx in skf.split(X_train_proc, y_train):
            clf = XGBClassifier(**params)
            clf.fit(X_train_proc[train_idx], y_train[train_idx])
            probs = clf.predict_proba(X_train_proc[val_idx])[:, 1]
            auc_scores.append(roc_auc_score(y_train[val_idx], probs))
        return float(np.mean(auc_scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Optuna Best Trial ROC-AUC: %.4f with params: %s", study.best_value, study.best_params)

    best_params = study.best_params.copy()
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["eval_metric"] = "logloss"
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    return best_params


def train_pipeline(
    data_path: str = "data/raw/credit_risk_dataset.csv",
    output_dir: str = "models",
    run_optuna: bool = True,
) -> Dict[str, Any]:
    """Execute complete training, tuning, benchmarking, and persistence pipeline.

    Args:
        data_path: Path to raw input dataset.
        output_dir: Directory where models and metrics are saved.
        run_optuna: Whether to execute Optuna tuning.

    Returns:
        Dict[str, Any]: Dictionary containing benchmark summary and holdout test results.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # 1. Prepare data
    logger.info("Initiating dataset split and feature engineering...")
    X_train, X_test, y_train, y_test = prepare_data(data_path)

    # Save processed splits for reproducibility
    X_train.assign(loan_status=y_train).to_csv("data/processed/train.csv", index=False)
    X_test.assign(loan_status=y_test).to_csv("data/processed/test.csv", index=False)

    # 2. Fit preprocessor
    logger.info("Fitting Scikit-Learn preprocessing pipeline...")
    preprocessor = create_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    # 3. Cross-validate candidate models
    benchmark_df, candidate_models = cross_validate_models(
        X_train_proc=X_train_proc,
        y_train=y_train.values,
        n_splits=5,
    )
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION BENCHMARK RESULTS (5-FOLD STRATIFIED)")
    print("=" * 80)
    print(benchmark_df.to_string(index=False))
    print("=" * 80 + "\n")

    # 4. Fit Baseline Logistic Regression (Traditional Credit Scorecard)
    baseline_lr = candidate_models["Logistic Regression (Baseline)"]
    baseline_lr.fit(X_train_proc, y_train.values)
    baseline_probs = baseline_lr.predict_proba(X_test_proc)[:, 1]
    baseline_test_metrics = evaluate_predictions(y_test.values, baseline_probs)
    joblib.dump(baseline_lr, os.path.join(output_dir, "logistic_baseline.joblib"))

    # 5. Hyperparameter Tuning for Production XGBoost
    scale_pos = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    if run_optuna:
        best_xgb_params = tune_xgboost_optuna(X_train_proc, y_train.values, scale_pos, n_trials=10)
        champion_xgb = XGBClassifier(**best_xgb_params)
    else:
        champion_xgb = candidate_models["XGBoost"]

    logger.info("Fitting Champion Tuned XGBoost on full training set...")
    champion_xgb.fit(X_train_proc, y_train.values)

    # 6. Evaluate Champion on Holdout Test Set
    xgb_test_probs = champion_xgb.predict_proba(X_test_proc)[:, 1]
    xgb_test_metrics = evaluate_predictions(y_test.values, xgb_test_probs)

    print("\n" + "=" * 80)
    print("CHAMPION MODEL (XGBOOST) EVALUATION ON UNSEEN TEST SET (3,000 cases)")
    print("=" * 80)
    for k, v in xgb_test_metrics.items():
        print(f"  {k:15s}: {v}")
    print("=" * 80 + "\n")

    # 7. Persist production artifacts
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    metrics_path = os.path.join(output_dir, "metrics.json")
    feature_names_path = os.path.join(output_dir, "feature_names.json")

    joblib.dump(champion_xgb, best_model_path)
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "champion_model": "XGBoost (Optuna Tuned)",
        "test_metrics": xgb_test_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "cv_benchmark": benchmark_df.to_dict(orient="records"),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    logger.info("Successfully persisted all production artifacts to %s", output_dir)
    return metadata


if __name__ == "__main__":
    train_pipeline()
