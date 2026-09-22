"""Production inference and explainability module for Credit Risk Evaluator.

This module provides the RiskEvaluator class to load persisted pipeline artifacts,
validate inference requests, generate default probabilities, assign risk ratings (A-D),
and compute SHAP local explanations with human-readable business descriptions.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure root package importability
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import shap

from src.data_processing import engineer_features

logger = logging.getLogger(__name__)

# Human-friendly translations for banking stakeholders
FEATURE_TRANSLATIONS: Dict[str, str] = {
    "person_age": "Idade do Proponente (anos)",
    "person_income": "Renda Anual (R$)",
    "person_emp_length": "Tempo de Emprego (anos)",
    "loan_amnt": "Valor do Empréstimo Solicitado (R$)",
    "loan_int_rate": "Taxa de Juros Anual (%)",
    "loan_percent_income": "Comprometimento da Renda (Loan/Income)",
    "cb_person_cred_hist_length": "Tempo de Histórico de Crédito (anos)",
    "cred_hist_to_age_ratio": "Maturidade Financeira Relativa",
    "debt_burden_index": "Índice de Carga Financeira",
    "person_home_ownership_OTHER": "Moradia: Outro",
    "person_home_ownership_OWN": "Moradia: Imóvel Próprio",
    "person_home_ownership_RENT": "Moradia: Alugado",
    "loan_intent_EDUCATION": "Finalidade: Educação",
    "loan_intent_HOMEIMPROVEMENT": "Finalidade: Reforma Residencial",
    "loan_intent_MEDICAL": "Finalidade: Despesas Médicas",
    "loan_intent_PERSONAL": "Finalidade: Despesas Pessoais",
    "loan_intent_VENTURE": "Finalidade: Empreendimento/Negócios",
    "cb_person_default_on_file_Y": "Histórico de Inadimplência Registrado",
}


def get_risk_tier(probability: float) -> Dict[str, str]:
    """Map default probability to credit risk grade and business recommendation.

    Threshold tiers based on Basel credit scoring benchmarks:
    - Tier A: p < 0.10 (Prime, auto-approved)
    - Tier B: 0.10 <= p < 0.25 (Standard, approved with standard monitoring)
    - Tier C: 0.25 <= p < 0.50 (Subprime / Cautionary, requires collateral or guarantor)
    - Tier D: p >= 0.50 (High Risk / Critical, rejection recommended)

    Args:
        probability: Model estimated default probability in range [0, 1].

    Returns:
        Dict[str, str]: Rating letter, description, color, and credit action.
    """
    if probability < 0.10:
        return {
            "grade": "A",
            "tier_name": "Baixo Risco (Prime)",
            "color": "#10B981",  # Emerald Green
            "badge_color": "green",
            "recommendation": "Aprovação Automática com taxas de juros competitivas.",
            "loss_provision": "Provisão mínima (PCLD ~1.5%)",
        }
    elif probability < 0.25:
        return {
            "grade": "B",
            "tier_name": "Risco Moderado (Regular)",
            "color": "#3B82F6",  # Blue
            "badge_color": "blue",
            "recommendation": "Aprovação Padrão sujeita à comprovação de renda.",
            "loss_provision": "Provisão moderada (PCLD ~5.0%)",
        }
    elif probability < 0.50:
        return {
            "grade": "C",
            "tier_name": "Alto Risco (Atenção)",
            "color": "#F59E0B",  # Amber/Yellow
            "badge_color": "orange",
            "recommendation": "Encaminhar para Comitê de Crédito / Exigir garantias reais.",
            "loss_provision": "Provisão elevada (PCLD ~20.0%)",
        }
    else:
        return {
            "grade": "D",
            "tier_name": "Risco Crítico (Subprime)",
            "color": "#EF4444",  # Crimson Red
            "badge_color": "red",
            "recommendation": "Recusa Recomendada. Alta probabilidade de default.",
            "loss_provision": "Provisão integral (PCLD > 50.0%)",
        }


# Garante o caminho absoluto para a raiz do repositório no Streamlit Cloud
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class RiskEvaluator:
    """Production credit risk evaluator wrapping preprocessor, model, and SHAP explainer."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        feature_names_path: Optional[str] = None,
    ) -> None:
        """Initialize the evaluator and load artifacts."""
        self.model_path = model_path or os.path.join(BASE_DIR, "models", "best_model.joblib")
        self.preprocessor_path = preprocessor_path or os.path.join(BASE_DIR, "models", "preprocessor.joblib")
        self.feature_names_path = feature_names_path or os.path.join(BASE_DIR, "models", "feature_names.json")

        self.model = None
        self.preprocessor = None
        self.feature_names: List[str] = []
        self.explainer: Optional[Any] = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model, preprocessor, and initialize SHAP explainer."""
        try:
            logger.info("Loading evaluator artifacts from models/ directory.")
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)

            if os.path.exists(self.feature_names_path):
                with open(self.feature_names_path, "r", encoding="utf-8") as f:
                    self.feature_names = json.load(f)

            # Initialize SHAP explainer according to model family
            model_type = type(self.model).__name__
            logger.info("Initializing SHAP explainer for %s", model_type)

            if "XGB" in model_type or "LGBM" in model_type or "Forest" in model_type:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback to Explainer for Linear/Logistic
                self.explainer = shap.Explainer(self.model)

            logger.info("Evaluator ready for online inference.")
        except Exception as exc:
            logger.error("Failed to load artifacts: %s", str(exc))
            raise RuntimeError(f"Could not load credit evaluator artifacts: {exc}") from exc

    def predict(self, input_data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate single or batch applicant data and return risk intelligence.

        Args:
            input_data: Applicant data dictionary or dataframe with raw features.

        Returns:
            Dict containing probability, rating tier, SHAP contributions, and business insights.
        """
        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
        else:
            df_input = input_data.copy()

        # 1. Feature Engineering
        df_feat = engineer_features(df_input)

        # 2. Pipeline Preprocessing
        X_proc = self.preprocessor.transform(df_feat)

        # 3. Model Inference (default probability)
        try:
            prob = float(self.model.predict_proba(X_proc)[0, 1])
        except Exception:
            prob = float(self.model.predict(X_proc)[0])

        prob = float(np.clip(prob, 0.0001, 0.9999))
        tier_info = get_risk_tier(prob)

        # 4. SHAP Feature Attribution
        shap_details = self._explain_instance(X_proc[0])

        return {
            "default_probability": round(prob, 4),
            "default_percentage": round(prob * 100, 2),
            "risk_grade": tier_info["grade"],
            "tier_name": tier_info["tier_name"],
            "tier_color": tier_info["color"],
            "recommendation": tier_info["recommendation"],
            "loss_provision": tier_info["loss_provision"],
            "shap_explanation": shap_details,
            "raw_features": df_feat.iloc[0].to_dict(),
        }

    def _explain_instance(self, processed_vector: np.ndarray) -> Dict[str, Any]:
        """Compute SHAP attribution for a single preprocessed observation vector.

        Args:
            processed_vector: 1D numpy array of transformed features.

        Returns:
            Dict containing base value, feature contributions, and top drivers.
        """
        try:
            vec_2d = processed_vector.reshape(1, -1)
            raw_shap = self.explainer(vec_2d)

            # Handle binary classification 3D or 2D array output
            if len(raw_shap.values.shape) == 3:
                vals = raw_shap.values[0, :, 1]
                base_val = float(raw_shap.base_values[0, 1])
            else:
                vals = raw_shap.values[0]
                base_val = float(raw_shap.base_values[0])

            features = self.feature_names if self.feature_names else [f"f_{i}" for i in range(len(vals))]

            impacts = []
            for feat, val, feat_val in zip(features, vals, processed_vector):
                label_pt = FEATURE_TRANSLATIONS.get(feat, feat)
                impacts.append({
                    "feature": feat,
                    "label": label_pt,
                    "shap_value": round(float(val), 4),
                    "feature_value": round(float(feat_val), 3),
                    "impact_direction": "Aumenta Risco" if val > 0 else "Reduz Risco",
                })

            # Sort by absolute SHAP impact
            impacts_sorted = sorted(impacts, key=lambda x: abs(x["shap_value"]), reverse=True)

            # Separate into top risk drivers and top protective factors
            risk_drivers = [item for item in impacts_sorted if item["shap_value"] > 0][:5]
            risk_mitigators = [item for item in impacts_sorted if item["shap_value"] < 0][:5]

            return {
                "base_value": round(base_val, 4),
                "all_features": impacts_sorted,
                "top_risk_drivers": risk_drivers,
                "top_risk_mitigators": risk_mitigators,
            }
        except Exception as exc:
            logger.warning("SHAP computation failed: %s. Returning fallback nulls.", str(exc))
            return {
                "base_value": 0.0,
                "all_features": [],
                "top_risk_drivers": [],
                "top_risk_mitigators": [],
            }


if __name__ == "__main__":
    evaluator = RiskEvaluator()
    sample_applicant = {
        "person_age": 28,
        "person_income": 45000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_amnt": 15000,
        "loan_int_rate": 14.5,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4,
    }
    result = evaluator.predict(sample_applicant)
    print("=" * 60)
    print("SAMPLE EVALUATION RESULT:")
    print("Default Probability:", f"{result['default_percentage']}%")
    print("Rating Grade:", result["risk_grade"], "-", result["tier_name"])
    print("Recommendation:", result["recommendation"])
    print("\nTop Risk Drivers:")
    for d in result["shap_explanation"]["top_risk_drivers"]:
        print(f"  + {d['label']}: SHAP = {d['shap_value']}")
    print("\nTop Mitigators:")
    for m in result["shap_explanation"]["top_risk_mitigators"]:
        print(f"  - {m['label']}: SHAP = {m['shap_value']}")
    print("=" * 60)
