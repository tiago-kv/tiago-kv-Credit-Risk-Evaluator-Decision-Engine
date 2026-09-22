"""Unit tests for Credit Risk Evaluator pipeline.

Tests data loading, feature engineering, model inference, SHAP explanations,
and edge case behavior using pytest.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add root project to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import (
    NUMERICAL_FEATURES,
    create_preprocessor,
    engineer_features,
    load_raw_data,
    prepare_data,
)
from src.predict import RiskEvaluator, get_risk_tier
from src.train import calculate_ks_statistic, evaluate_predictions


@pytest.fixture(scope="session")
def raw_data_path() -> str:
    """Fixture providing path to the raw dataset."""
    return "data/raw/credit_risk_dataset.csv"


@pytest.fixture(scope="session")
def sample_applicant_prime() -> dict:
    """Fixture for a low-risk prime applicant."""
    return {
        "person_age": 45,
        "person_income": 120000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 15.0,
        "loan_intent": "HOMEIMPROVEMENT",
        "loan_amnt": 10000,
        "loan_int_rate": 7.5,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 12,
    }


@pytest.fixture(scope="session")
def sample_applicant_subprime() -> dict:
    """Fixture for a high-risk subprime applicant."""
    return {
        "person_age": 22,
        "person_income": 18000,
        "person_home_ownership": "RENT",
        "person_emp_length": 0.5,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_amnt": 15000,
        "loan_int_rate": 21.0,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 2,
    }


def test_data_loading_and_shape(raw_data_path: str):
    """Test raw dataset loading and schema integrity."""
    df = load_raw_data(raw_data_path)
    assert not df.empty
    assert len(df) >= 1000
    assert "loan_status" in df.columns
    assert set(df["loan_status"].unique()).issubset({0, 1})


def test_feature_engineering():
    """Test engineered features creation and domain constraints."""
    df_sample = pd.DataFrame([{
        "person_age": 30,
        "person_income": 60000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "PERSONAL",
        "loan_amnt": 12000,
        "loan_int_rate": 10.0,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 6,
    }])
    df_feat = engineer_features(df_sample)

    assert "loan_percent_income" in df_feat.columns
    assert "cred_hist_to_age_ratio" in df_feat.columns
    assert "debt_burden_index" in df_feat.columns

    # Check calculated values
    expected_ratio = round(12000 / 60000, 4)
    assert df_feat["loan_percent_income"].iloc[0] == expected_ratio
    assert 0.0 <= df_feat["cred_hist_to_age_ratio"].iloc[0] <= 1.0


def test_preprocessor_pipeline(raw_data_path: str):
    """Test ColumnTransformer fitting and transform dimensions."""
    X_train, X_test, _, _ = prepare_data(raw_data_path)
    preprocessor = create_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    assert X_train_proc.shape[0] == len(X_train)
    assert X_test_proc.shape[0] == len(X_test)
    assert X_train_proc.shape[1] == X_test_proc.shape[1]
    assert not np.isnan(X_train_proc).any()


def test_ks_statistic_computation():
    """Test Kolmogorov-Smirnov statistic calculation with synthetic distributions."""
    # Perfect separation -> KS = 1.0
    y_true_perfect = np.array([0, 0, 0, 1, 1, 1])
    y_prob_perfect = np.array([0.1, 0.2, 0.15, 0.85, 0.9, 0.95])
    ks_perfect = calculate_ks_statistic(y_true_perfect, y_prob_perfect)
    assert ks_perfect == 1.0

    # Random guesses -> KS close to 0.0
    y_true_rand = np.array([0, 1, 0, 1, 0, 1])
    y_prob_rand = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ks_rand = calculate_ks_statistic(y_true_rand, y_prob_rand)
    assert ks_rand == 0.0


def test_risk_tier_mapping():
    """Test risk grade mapping and threshold boundaries."""
    tier_a = get_risk_tier(0.05)
    assert tier_a["grade"] == "A"

    tier_b = get_risk_tier(0.18)
    assert tier_b["grade"] == "B"

    tier_c = get_risk_tier(0.35)
    assert tier_c["grade"] == "C"

    tier_d = get_risk_tier(0.75)
    assert tier_d["grade"] == "D"


def test_inference_pipeline_predictions(
    sample_applicant_prime: dict,
    sample_applicant_subprime: dict,
):
    """Test RiskEvaluator end-to-end inference and SHAP explainability."""
    evaluator = RiskEvaluator()

    # Prime applicant test
    res_prime = evaluator.predict(sample_applicant_prime)
    assert "default_probability" in res_prime
    assert 0.0 <= res_prime["default_probability"] <= 1.0
    assert res_prime["risk_grade"] in ["A", "B", "C", "D"]
    assert "shap_explanation" in res_prime

    # Subprime applicant test
    res_subprime = evaluator.predict(sample_applicant_subprime)
    assert 0.0 <= res_subprime["default_probability"] <= 1.0

    # Subprime default probability should be strictly greater than Prime
    assert res_subprime["default_probability"] > res_prime["default_probability"]

    # Verify SHAP drivers structure
    shap_info = res_subprime["shap_explanation"]
    assert "top_risk_drivers" in shap_info
    assert "top_risk_mitigators" in shap_info
    assert len(shap_info["all_features"]) > 0
