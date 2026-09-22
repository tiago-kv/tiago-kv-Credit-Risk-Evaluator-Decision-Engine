"""Interactive Credit Risk Assessment and SHAP Explainability Web App.

Developed for Portfolio: Credit Risk Evaluator.
Powered by XGBoost, SHAP, Scikit-Learn, and Streamlit.
"""

import json
import os
import sys
from typing import Any, Dict

# Ensure project root is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import RiskEvaluator

# Page configuration
st.set_page_config(
    page_title="Credit Risk Evaluator | ML Decision Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for financial dashboard aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-green { background-color: #D1FAE5; color: #065F46; }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-orange { background-color: #FEF3C7; color: #92400E; }
    .badge-red { background-color: #FEE2E2; color: #991B1B; }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        height: 3rem;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_risk_evaluator() -> RiskEvaluator:
    """Cache and return the RiskEvaluator instance to optimize latency."""
    return RiskEvaluator()


@st.cache_data(show_spinner=False)
def load_model_metrics() -> Dict[str, Any]:
    """Load benchmark and test metrics JSON."""
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def create_gauge_chart(probability_pct: float, risk_color: str) -> go.Figure:
    """Generate a modern Plotly Gauge chart for default probability."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_pct,
            number={"suffix": "%", "font": {"size": 42, "color": "#1E293B"}},
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": "<b>Probabilidade de Default Estimada</b><br><span style='font-size:0.8em;color:#64748B'>P(Inadimplência em 12 meses)</span>",
                "font": {"size": 16},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": risk_color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 10], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [10, 25], "color": "rgba(59, 130, 246, 0.25)"},
                    {"range": [25, 50], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [50, 100], "color": "rgba(239, 68, 68, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#1E293B", "width": 4},
                    "thickness": 0.8,
                    "value": probability_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=320,
        margin={"t": 60, "b": 20, "l": 30, "r": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def create_shap_waterfall_chart(shap_info: Dict[str, Any]) -> go.Figure:
    """Generate a horizontal Plotly bar chart depicting SHAP impact factors."""
    features = shap_info.get("all_features", [])
    if not features:
        return go.Figure()

    # Take top 8 contributors by magnitude
    top_items = features[:8]
    # Invert order for ascending horizontal display
    top_items = top_items[::-1]

    labels = [item["label"] for item in top_items]
    shap_vals = [item["shap_value"] for item in top_items]
    colors = ["#EF4444" if val > 0 else "#10B981" for val in shap_vals]
    texts = [f"+{val:.3f} (Risco)" if val > 0 else f"{val:.3f} (Proteção)" for val in shap_vals]

    fig = go.Figure(
        go.Bar(
            x=shap_vals,
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"width": 1, "color": "#E2E8F0"}},
            text=texts,
            textposition="auto",
        )
    )

    fig.update_layout(
        title={
            "text": "<b>Explicabilidade SHAP: Fatores Determinantes da Decisão</b><br><span style='font-size:0.8em;color:#64748B'>Impacto Marginal no Log-Odds de Inadimplência</span>",
            "font": {"size": 15},
        },
        xaxis_title="Contribuição SHAP (Margem de Risco)",
        yaxis_title="",
        height=360,
        margin={"t": 60, "b": 40, "l": 180, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
        xaxis={"zeroline": True, "zerolinecolor": "#94A3B8", "zerolinewidth": 1.5},
        font={"family": "Inter, sans-serif"},
    )
    return fig


def main() -> None:
    """Main application routine."""
    evaluator = get_risk_evaluator()
    metrics = load_model_metrics()

    # Sidebar: Proponent Configuration & Presets
    st.sidebar.image("https://img.icons8.com/color/96/000000/bank-cards.png", width=64)
    st.sidebar.title("Simulador de Crédito")
    st.sidebar.caption("Motor de Decisão & Rating de Risco | XGBoost + SHAP")

    # Quick Presets for Demo
    preset = st.sidebar.selectbox(
        "Carregar Exemplo Pré-definido:",
        [
            "Customizado (Inserir Manualmente)",
            "Perfil A - Prime (Excelente)",
            "Perfil B - Risco Médio (Padrão)",
            "Perfil C - Alavancado (Atenção)",
            "Perfil D - Subprime (Inadimplência Alta)",
        ],
    )

    # Preset values
    if preset == "Perfil A - Prime (Excelente)":
        d_age, d_income, d_home, d_emp = 45, 130000, "MORTGAGE", 14.0
        d_intent, d_loan, d_rate, d_default, d_cred = "HOMEIMPROVEMENT", 12000, 7.8, "N", 12
    elif preset == "Perfil B - Risco Médio (Padrão)":
        d_age, d_income, d_home, d_emp = 32, 65000, "RENT", 4.0
        d_intent, d_loan, d_rate, d_default, d_cred = "PERSONAL", 14000, 11.2, "N", 6
    elif preset == "Perfil C - Alavancado (Atenção)":
        d_age, d_income, d_home, d_emp = 28, 42000, "RENT", 2.0
        d_intent, d_loan, d_rate, d_default, d_cred = "DEBTCONSOLIDATION", 18000, 15.5, "N", 4
    elif preset == "Perfil D - Subprime (Inadimplência Alta)":
        d_age, d_income, d_home, d_emp = 23, 22000, "RENT", 0.8
        d_intent, d_loan, d_rate, d_default, d_cred = "DEBTCONSOLIDATION", 16000, 21.0, "Y", 2
    else:
        d_age, d_income, d_home, d_emp = 30, 55000, "RENT", 3.5
        d_intent, d_loan, d_rate, d_default, d_cred = "PERSONAL", 12000, 12.0, "N", 5

    with st.sidebar.form("credit_form"):
        st.subheader("1. Dados do Proponente")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            age = st.number_input("Idade (anos)", min_value=18, max_value=85, value=d_age)
            home_ownership = st.selectbox(
                "Situação Moradia",
                ["RENT", "OWN", "MORTGAGE", "OTHER"],
                index=["RENT", "OWN", "MORTGAGE", "OTHER"].index(d_home),
            )
        with col_s2:
            income = st.number_input("Renda Anual (R$)", min_value=5000, max_value=1000000, value=d_income, step=5000)
            emp_length = st.number_input("Tempo de Emprego (anos)", min_value=0.0, max_value=50.0, value=float(d_emp), step=0.5)

        st.subheader("2. Dados da Operação")
        col_s3, col_s4 = st.columns(2)
        with col_s3:
            loan_intent = st.selectbox(
                "Finalidade",
                ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
                index=["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"].index(d_intent),
            )
            loan_amnt = st.number_input("Valor Solicitado (R$)", min_value=1000, max_value=100000, value=d_loan, step=1000)
        with col_s4:
            loan_int_rate = st.number_input("Taxa de Juros Anual (%)", min_value=3.0, max_value=35.0, value=float(d_rate), step=0.25)
            cred_hist = st.number_input("Histórico de Crédito (anos)", min_value=1, max_value=40, value=d_cred)

        default_on_file = st.selectbox(
            "Histórico Prévio de Inadimplência?",
            ["N", "Y"],
            index=["N", "Y"].index(d_default),
            help="Registro prévio no Bureau de Crédito (ex: Serasa/SPC)",
        )

        submit_btn = st.form_submit_button("⚡ Avaliar Risco de Crédito")

    # Header section
    st.markdown('<div class="main-title">Credit Risk Evaluator & Decision Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Avaliação inteligente de risco de crédito com modelos Gradient Boosting calibrados, explicabilidade individual via SHAP e métricas de Basileia.</div>',
        unsafe_allow_html=True,
    )

    # Input payload
    applicant_payload = {
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home_ownership,
        "person_emp_length": emp_length,
        "loan_intent": loan_intent,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "cb_person_default_on_file": default_on_file,
        "cb_person_cred_hist_length": cred_hist,
    }

    # Execute prediction
    result = evaluator.predict(applicant_payload)
    prob_pct = result["default_percentage"]
    grade = result["risk_grade"]
    badge_color = result["tier_color"]
    loan_pct_income = (loan_amnt / max(income, 1)) * 100

    badge_class_map = {
        "A": "badge-green",
        "B": "badge-blue",
        "C": "badge-orange",
        "D": "badge-red",
    }
    badge_css = badge_class_map.get(grade, "badge-blue")

    # Top KPI Metrics Cards
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    with kpi_c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="color:#64748B;font-size:0.85rem;font-weight:600;">PROBABILIDADE DE DEFAULT</span>
                <h2 style="margin:0.2rem 0;color:#1E293B;">{prob_pct:.2f}%</h2>
                <span style="color:#64748B;font-size:0.8rem;">Benchmark Aceitável: &lt; 25%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="color:#64748B;font-size:0.85rem;font-weight:600;">CLASSIFICAÇÃO DE RISCO</span>
                <div style="margin:0.3rem 0;"><span class="badge {badge_css}" style="font-size:1.1rem;padding:6px 16px;">RATING {grade}</span></div>
                <span style="color:#64748B;font-size:0.8rem;">{result['tier_name']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="color:#64748B;font-size:0.85rem;font-weight:600;">RECOMENDAÇÃO POLÍTICA</span>
                <h4 style="margin:0.3rem 0;color:#1E293B;font-size:1rem;">{'APROVAR' if grade in ['A', 'B'] else 'COMITÊ / RECUSA'}</h4>
                <span style="color:#64748B;font-size:0.8rem;">{result['recommendation']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="color:#64748B;font-size:0.85rem;font-weight:600;">COMPROMETIMENTO RENDA</span>
                <h2 style="margin:0.2rem 0;color:#1E293B;">{loan_pct_income:.1f}%</h2>
                <span style="color:#64748B;font-size:0.8rem;">Limite Recomendado: &lt; 35%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations Row: Gauge & SHAP Waterfall
    col_vis1, col_vis2 = st.columns([1, 1.25])
    with col_vis1:
        gauge_fig = create_gauge_chart(prob_pct, badge_color)
        st.plotly_chart(gauge_fig, use_container_width=True)

        st.info(
            f"**Diretriz Regulatória (PCLD):** {result['loss_provision']}\n\n"
            f"**Ação Recomendada:** {result['recommendation']}"
        )

    with col_vis2:
        shap_fig = create_shap_waterfall_chart(result["shap_explanation"])
        st.plotly_chart(shap_fig, use_container_width=True)

    st.markdown("---")

    # Tabs: Deep Dive Analysis & Architecture
    tab1, tab2, tab3 = st.tabs([
        "🔍 Detalhamento dos Fatores SHAP",
        "🎯 Simulador What-If (Mitigação de Risco)",
        "📊 Métricas & Benchmark de Modelos",
    ])

    with tab1:
        st.subheader("Auditoria de Decisão & Fatores Determinantes (SHAP Local)")
        st.write(
            "Em conformidade com as exigências do Banco Central e as melhores práticas de Governança de IA (LGPD / Fair Lending), "
            "cada predição é auditável. Abaixo estão os fatores que mais influenciaram esta concessão:"
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("##### 🚨 Principais Vetores de Risco (Aumentam Probabilidade de Inadimplência)")
            drivers = result["shap_explanation"]["top_risk_drivers"]
            if drivers:
                for d in drivers:
                    st.error(f"**{d['label']}**: Impacto SHAP = `+{d['shap_value']:.4f}`")
            else:
                st.success("Nenhum fator de risco severo detectado no perfil.")

        with col_d2:
            st.markdown("##### 🛡️ Principais Fatores Mitigadores (Reduzem o Risco / Protetivos)")
            mitigators = result["shap_explanation"]["top_risk_mitigators"]
            if mitigators:
                for m in mitigators:
                    st.success(f"**{m['label']}**: Impacto SHAP = `{m['shap_value']:.4f}`")
            else:
                st.warning("Poucos fatores mitigadores encontrados.")

    with tab2:
        st.subheader("Simulador 'What-If': Como Tornar Esta Proposta Aprovável?")
        st.write(
            "Ferramenta para a mesa de crédito renegociar as condições comerciais do contrato e adequar o perfil à política interna de apetite a risco:"
        )

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            sim_loan = st.slider(
                "Reduzir Valor do Empréstimo (R$)",
                min_value=1000,
                max_value=int(loan_amnt),
                value=int(loan_amnt * 0.7),
                step=500,
            )
            sim_rate = st.slider(
                "Ajustar Taxa de Juros (%)",
                min_value=5.0,
                max_value=30.0,
                value=float(loan_int_rate),
                step=0.5,
            )
        with sim_c2:
            sim_payload = applicant_payload.copy()
            sim_payload["loan_amnt"] = sim_loan
            sim_payload["loan_int_rate"] = sim_rate
            sim_res = evaluator.predict(sim_payload)

            delta_prob = sim_res["default_percentage"] - prob_pct
            st.metric(
                label="Nova Probabilidade de Default",
                value=f"{sim_res['default_percentage']:.2f}%",
                delta=f"{delta_prob:.2f}%",
                delta_color="inverse",
            )
            st.write(f"**Novo Rating Simulado:** Rating {sim_res['risk_grade']} ({sim_res['tier_name']})")
            st.write(f"**Novo Parecer:** {sim_res['recommendation']}")

    with tab3:
        st.subheader("Benchmark de Modelos & Métricas de Validação")
        st.write(
            "Resultados comparativos obtidos através de validação cruzada estratificada em 5 folds (StratifiedKFold) "
            "e avaliação final em conjunto de teste cego (3.000 amostras):"
        )

        if metrics and "cv_benchmark" in metrics:
            df_cv = pd.DataFrame(metrics["cv_benchmark"])
            st.dataframe(df_cv.style.highlight_max(subset=["ROC-AUC (Mean)", "KS Statistic"], color="#D1FAE5"), use_container_width=True)

        if metrics and "test_metrics" in metrics:
            st.markdown("##### Desempenho do Modelo Campeão (XGBoost) no Test Set:")
            t_m = metrics["test_metrics"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("ROC-AUC", f"{t_m.get('roc_auc', 0):.4f}")
            m2.metric("Estatística KS", f"{t_m.get('ks_statistic', 0):.4f}")
            m3.metric("F1-Score", f"{t_m.get('f1_score', 0):.4f}")
            m4.metric("Precision", f"{t_m.get('precision', 0):.4f}")
            m5.metric("Recall", f"{t_m.get('recall', 0):.4f}")

        st.info(
            "💡 **Por que o KS Statistic (Kolmogorov-Smirnov)?**\n\n"
            "Em modelagem de risco bancário (padrão Basileia II/III), a métrica KS avalia a capacidade máxima do modelo de separar "
            "a curva acumulada de bons pagadores da curva de maus pagadores. Valores de KS acima de 0.40 são considerados bons; "
            "valores acima de 0.60 indicam excelente capacidade discriminatória da esteira de crédito."
        )


if __name__ == "__main__":
    main()
