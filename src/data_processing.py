"""Data processing and feature engineering module for Credit Risk Evaluator.

This module provides end-to-end data loading, schema validation,
feature engineering, and preprocessor pipeline creation for credit risk models.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Configure module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Expected schema definitions
NUMERICAL_FEATURES: List[str] = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "cred_hist_to_age_ratio",
    "debt_burden_index",
]

RAW_NUMERICAL_FEATURES: List[str] = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_FEATURES: List[str] = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]

TARGET_COLUMN: str = "loan_status"


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load credit dataset from CSV file.

    Args:
        file_path: Path to the CSV raw data file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is empty or corrupted.
    """
    logger.info("Loading raw dataset from %s", file_path)
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"File at {file_path} is empty.")
        logger.info("Successfully loaded %d rows and %d columns.", df.shape[0], df.shape[1])
        return df
    except FileNotFoundError:
        logger.error("Dataset not found at path: %s", file_path)
        raise
    except Exception as exc:
        logger.error("Unexpected error loading data: %s", str(exc))
        raise


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering relevant to credit risk scoring.

    Features generated:
    1. loan_percent_income: Recalculated/validated loan amount relative to income.
    2. cred_hist_to_age_ratio: Ratio of credit bureau history length to adult age (age - 18).
    3. debt_burden_index: Interaction between interest rate and loan percent income.

    Args:
        df: Input dataframe containing raw credit features.

    Returns:
        pd.DataFrame: Dataframe with engineered features.
    """
    logger.info("Applying domain-specific feature engineering.")
    df_feat = df.copy()

    # Protect against division by zero
    safe_income = np.where(df_feat["person_income"] <= 0, 1.0, df_feat["person_income"])
    df_feat["loan_percent_income"] = np.round(df_feat["loan_amnt"] / safe_income, 4)

    # Ratio of credit history length to adult life span
    adult_years = np.maximum(df_feat["person_age"] - 18, 1)
    df_feat["cred_hist_to_age_ratio"] = np.round(
        df_feat["cb_person_cred_hist_length"] / adult_years, 4
    )
    df_feat["cred_hist_to_age_ratio"] = np.clip(df_feat["cred_hist_to_age_ratio"], 0.0, 1.0)

    # Debt burden index: captures compounded risk of high interest on high principal relative to income
    df_feat["debt_burden_index"] = np.round(
        (df_feat["loan_int_rate"].fillna(df_feat["loan_int_rate"].median()) / 100.0)
        * df_feat["loan_percent_income"],
        4,
    )

    return df_feat


def create_preprocessor() -> ColumnTransformer:
    """Create Scikit-Learn ColumnTransformer pipeline.

    Numerical pipeline:
    - Median imputation for missing values (e.g. emp_length, int_rate)
    - RobustScaler to be robust against income/loan outliers

    Categorical pipeline:
    - Frequent category imputation
    - OneHotEncoder with handle_unknown='ignore'

    Returns:
        ColumnTransformer: Configured preprocessor.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract transformed feature names from fitted ColumnTransformer.

    Args:
        preprocessor: Fitted ColumnTransformer instance.

    Returns:
        List[str]: Names of all output features.
    """
    output_features: List[str] = []
    for name, pipe, cols in preprocessor.transformers_:
        if name == "num":
            output_features.extend(cols)
        elif name == "cat":
            encoder = pipe.named_steps["onehot"]
            cat_features = encoder.get_feature_names_out(cols).tolist()
            output_features.extend(cat_features)
    return output_features


def prepare_data(
    file_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Complete preparation workflow: load, engineer features, and train/test split.

    Args:
        file_path: Path to raw dataset CSV.
        test_size: Proportion of test sample. Default 0.2.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple containing X_train, X_test, y_train, y_test.
    """
    df_raw = load_raw_data(file_path)
    df_eng = engineer_features(df_raw)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df_eng[feature_cols]
    y = df_eng[TARGET_COLUMN]

    logger.info("Splitting dataset into train (%.0f%%) and test (%.0f%%).", (1 - test_size) * 100, test_size * 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)
    logger.info("Target distribution - Train: %.2f%% positive, Test: %.2f%% positive",
                y_train.mean() * 100, y_test.mean() * 100)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = prepare_data("data/raw/credit_risk_dataset.csv")
    prep = create_preprocessor()
    X_tr_proc = prep.fit_transform(X_tr)
    feat_names = get_feature_names(prep)
    print("Preprocessing test successful!")
    print(f"Processed shape: {X_tr_proc.shape}")
    print(f"Feature count: {len(feat_names)}")
