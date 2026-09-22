# 💳 Credit Risk Evaluator & Decision Engine

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Live-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Champion-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)](https://shap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-pytest%20passing-brightgreen.svg)](tests/)

> **Motor preditivo de análise de risco de crédito com explicabilidade individual via SHAP, calibração de probabilidades de inadimplência (Probability of Default - PD), classificação em ratings bancários (A-D) e simulador de políticas de crédito em tempo real.**

---

## 📌 Contexto de Negócio & Motivação Econômica

Em instituições financeiras e fintechs, a precificação do risco de crédito é o pilar central da rentabilidade e da solvência. Uma esteira de crédito ineficiente acarreta dois tipos graves de perdas:
1. **Falso Negativo (Risco de Inadimplência Não Detectado):** Conceder crédito a um tomador que entra em default causa perda integral ou parcial do principal, onerando a Provisão para Créditos de Liquidação Duvidosa (**PCLD**) e consumindo capital regulatório (**Basileia II / III**).
2. **Falso Positivo (Recusa Indevida de Bom Pagador):** Negar crédito a um cliente rentável gera custo de oportunidade comercial (*churn* para concorrentes).

Tradicionalmente, bancos utilizavam **Scorecards lineares baseados em Regressão Logística** com WoE (*Weight of Evidence*). Embora explicáveis, modelos puramente lineares perdem interações complexas (por exemplo: como a taxa de juros interage com o comprometimento de renda sob diferentes maturidades financeiras).

Este projeto desenvolve uma solução ponta a ponta que une a **alta capacidade discriminatória de algoritmos Gradient Boosting (XGBoost/LightGBM)** com a **explicabilidade matemática axiomática dos valores de SHAP (Shapley Additive exPlanations)**, cumprindo os padrões de auditoria do **Banco Central do Brasil (BACEN)** e o direito à explicação previsto na **LGPD (Lei Geral de Proteção de Dados)**.

---

## 🏗️ Arquitetura e Estrutura do Repositório

```text
credit-risk-evaluator/
│
├── data/
│   ├── raw/                 # Dataset bruto (ignorado pelo git)
│   └── processed/           # Splits tratados train/test (ignorado pelo git)
│
├── notebooks/
│   ├── 01_eda.ipynb         # Análise exploratória, desbalanceamento, correlações
│   ├── 02_model_training.ipynb  # Treinamento, StratifiedKFold e Optuna tuning
│   └── 03_evaluation.ipynb  # ROC, KS Statistic, Confusion Matrix e SHAP Global/Local
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # Pipeline modular de features e ColumnTransformer
│   ├── train.py             # Script de treinamento, cross-validation e persistência
│   └── predict.py           # Classe RiskEvaluator, inferência e SHAP explainers
│
├── app/
│   └── app.py               # Web App interativo Streamlit com gráficos Plotly
│
├── models/
│   ├── best_model.joblib    # Modelo campeão serializado (XGBoost tunado)
│   ├── preprocessor.joblib  # Pipeline de transformação (imputação + scaling + OHE)
│   ├── feature_names.json   # Metadados das variáveis processadas
│   └── metrics.json         # Métricas de validação cruzada e teste
│
├── tests/
│   └── test_pipeline.py     # Testes unitários automatizados com pytest
│
├── requirements.txt         # Dependências do projeto
├── .gitignore               # Configurações de exclusão de artefatos pesados
└── README.md                # Documentação técnica completa
```

---

## 🔬 Metodologia e Engenharia de Features

### 1. Tratamento de Dados & Pré-processamento
- **Imputação Robusta:** Variáveis numéricas como `loan_int_rate` (~6% nulos) e `person_emp_length` (~4% nulos) receberam imputação pela mediana para evitar viés introduzido por médias sensíveis a outliers.
- **Escalonamento:** Aplicação de `RobustScaler` nas variáveis financeiras (`person_income`, `loan_amnt`), preservando as distâncias relativas sem distorção por rendas milionárias de cauda longa.
- **Codificação Categórica:** `OneHotEncoder(drop='first', handle_unknown='ignore')` para tratar `person_home_ownership`, `loan_intent` e `cb_person_default_on_file`.

### 2. Features Derivadas (Domínio Financeiro / Econômico)
- **`loan_percent_income`:** Relação Empréstimo / Renda Anual (medida primária de alavancagem financeira).
- **`cred_hist_to_age_ratio`:** Razão entre anos de histórico no bureau e idade adulta ativa (`age - 18`), capturando maturidade financeira do proponente.
- **`debt_burden_index`:** Índice de carga financeira que multiplica o comprometimento de renda pela taxa de juros anualizada, capturando o peso do serviço da dívida.

### 3. Tratamento de Desbalanceamento
A base apresenta ~25.8% de proponentes inadimplentes (desbalanceamento típico de concessão de crédito). Foi utilizado o parâmetro `scale_pos_weight` calculado dinamicamente ($\frac{N_{neg}}{N_{pos}}$) nos modelos de Gradient Boosting e `class_weight='balanced'` na Regressão Logística e Random Forest.

---

## 📊 Resultados e Benchmark de Modelos

Os modelos foram avaliados via **Stratified 5-Fold Cross-Validation** e submetidos a um teste cego final com **3.000 amostras out-of-sample**:

### Validação Cruzada Estratificada (5 Folds)

| Modelo | ROC-AUC Médio | Desvio Padrão | KS Statistic | F1-Score | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** 🏆 | **0.9071** | **±0.0058** | **0.6652** | **0.7310** | **0.7850** | **0.6840** |
| **Logistic Regression (Scorecard)** | 0.9095 | ±0.0065 | 0.6723 | 0.7255 | 0.8108 | 0.6567 |
| **Random Forest** | 0.9022 | ±0.0078 | 0.6570 | 0.7355 | 0.7375 | 0.7339 |
| **LightGBM** | 0.8995 | ±0.0062 | 0.6531 | 0.7197 | 0.7724 | 0.6741 |

### Desempenho no Holdout Test Set (3.000 Casos)

- **ROC-AUC:** `0.8890` (Excelente discriminação global)
- **KS Statistic (Kolmogorov-Smirnov):** `0.6377` (Métrica padrão Basileia; benchmark regulatório > 0.40)
- **Acurácia Global:** `83.83%`
- **F1-Score:** `0.7104`
- **Recall da Classe Default:** `76.97%` (captura quase 8 em cada 10 inadimplentes reais)
- **Precision:** `65.96%`

---

## 🧠 Explicação Técnica das Escolhas de Engenharia e ML

### 1. Por que XGBoost e não apenas Random Forest ou Regressão Logística?
1. **Modelagem de Não-Linearidades e Interações Complexas:** Árvores com gradiente capturam superfícies de decisão altamente não-lineares (como o aumento exponencial do risco quando o comprometimento de renda ultrapassa 40% combinado com juros altos).
2. **Otimização Direta da Função de Perda:** Ao contrário do Random Forest (que combina árvores independentes por média), o XGBoost minimiza uma função de perda regularizada ($L_1$ e $L_2$), reduzindo ativamente o viés residual em cada iteração e mitigando overfitting.
3. **Maturidade e Eficiência em Produção:** O XGBoost permite inferência em milissegundos via `xgboost.Booster` e integração nativa com o algoritmo *TreeSHAP*, executando em tempo polinomial $O(TLD^2)$.

### 2. Por que SHAP (SHapley Additive exPlanations)?
Métodos tradicionais como *Feature Importance por Gini* ou *Permutation Importance* sofrem de severas limitações:
- O índice de Gini superestima a relevância de variáveis contínuas com alta cardinalidade.
- A importância por permutação quebra a estrutura de correlação entre variáveis econômicas interligadas.
- **Vantagem Axiomática do SHAP:** Baseado na Teoria dos Jogos Cooperativos de Lloyd Shapley (Prêmio Nobel de Economia), o SHAP é o **único método** que garante consistência e aditividade local. A soma dos valores SHAP de cada feature somada ao valor base resulta exatamente no log-odds da predição individual:

$$\ln\left(\frac{P(\text{Default})}{1 - P(\text{Default})}\right) = \phi_0 + \sum_{j=1}^{M} \phi_j$$

Isso permite explicar ao cliente ou ao regulador exatamente o motivo de cada recusa (ex: "+0.22 devido ao comprometimento de renda acima de 45%").

### 3. Por que a Estatística KS (Kolmogorov-Smirnov)?
Em risco de crédito bancário, a ROC-AUC avalia o ranqueamento geral, mas o **KS Statistic** avalia a **distância máxima entre a função de distribuição acumulada (CDF) dos adimplentes versus inadimplentes**:

$$KS = \max_s |F_{\text{inadimplente}}(s) - F_{\text{adimplente}}(s)|$$

Um KS de **0.6377** comprova que o modelo separa com extrema eficácia a massa de tomadores saudáveis dos tomadores propensos ao default.

---

## 🎯 Política de Decisão e Classificação de Ratings (A, B, C, D)

O modelo traduz probabilidades contínuas em réguas de crédito acionáveis para as mesas operacionais:

| Rating | Faixa de Probabilidade ($P$) | Classificação de Risco | Recomendação de Política | Provisão PCLD Estimada |
| :---: | :---: | :---: | :---: | :---: |
| **A** | $0.0\% \le P < 10.0\%$ | Baixo Risco (Prime) | **Aprovação Automática**, concessão de taxas promocionais | ~1.5% |
| **B** | $10.0\% \le P < 25.0\%$ | Risco Moderado (Regular) | **Aprovação Padrão**, comprovação regular de renda | ~5.0% |
| **C** | $25.0\% \le P < 50.0\%$ | Alto Risco (Atenção) | **Encaminhar para Comitê**, exigir avalista ou colateral real | ~20.0% |
| **D** | $P \ge 50.0\%$ | Risco Crítico (Subprime) | **Recusa Recomendada**, alta probabilidade de perda de principal | > 50.0% |

---

## 💻 Como Reproduzir Este Projeto

### 1. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/credit-risk-evaluator.git
cd credit-risk-evaluator
```

### 2. Criar e Ativar Ambiente Virtual
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4. Executar os Testes Unitários
```bash
pytest tests/ -v
```

### 5. Retreinar o Pipeline Completo (Opcional)
```bash
python3 src/train.py
```

### 6. Executar o Web App Interativo (Streamlit)
```bash
streamlit run app/app.py
```
O app abrirá no navegador em `http://localhost:8501`.

---

## 🎥 Demonstração do App Interativo

O aplicativo Streamlit foi desenhado seguindo premissas de design financeiro de nível bancário:
- **Painel de Controle Lateral:** Seleção rápida de perfis pré-definidos (Prime, Moderado, Subprime) ou ajuste manual de parâmetros econômicos.
- **Gauge Chart Interativo (Plotly):** Mostrador de velocímetro com faixas de risco (Verde, Azul, Amarelo, Vermelho).
- **Cartões de KPI:** Probabilidade de default, Rating (A-D), Parecer de Política e Comprometimento de Renda.
- **Gráfico SHAP Waterfall:** Decomposição visual dos fatores que elevaram o risco (vermelho) ou atenuaram o risco (verde).
- **Simulador What-If:** Permite simular contrapropostas comerciais (ex: "Se reduzirmos o empréstimo em 30%, o cliente se torna aprovável?").

> **Como gravar o GIF para o GitHub:**
> 1. Inicie o app com `streamlit run app/app.py`.
> 2. Use o software [ScreenToGif](https://www.screentogif.com/) (Windows) ou [Kap](https://getkap.co/) (macOS/Linux).
> 3. Selecione a janela do navegador, clique em "Avaliar Risco", teste a troca de perfis e o simulador What-If.
> 4. Salve como `demo.gif` na pasta `assets/` e adicione ao topo do README: `![Demo App](assets/demo.gif)`.

---

## 🚀 Próximos Passos & Melhorias Futuras

1. **Modelagem de Perda Dado o Default (LGD) e Exposição no Default (EAD):** Integrar modelos para estimar a Perda Esperada completa:
   $$\text{Expected Loss (EL)} = \text{PD} \times \text{LGD} \times \text{EAD}$$
2. **Monitoramento de Data Drift com Evidently AI / Great Expectations:** Monitorar desvio de covariáveis na produção para acionar retreino automático da esteira.
3. **Containerização Docker & CI/CD com GitHub Actions:** Criar imagem Docker padronizada e pipeline automatizado com execução de testes no pull request.
4. **Deploy na Nuvem (Azure App Service / AWS ECS):** Disponibilizar a API FastAPI e o frontend Streamlit para consumo de sistemas legados.

---

## 👨‍💻 Autor

**Cientista de Dados & Economista**  
- Formação: Economia (FESP) | MBA em Data Science & Analytics (USP)  
- Especialidades: Machine Learning, Credit Risk Scoring, Análise de Dados, Azure, Microsoft Fabric e Power BI.  
- LinkedIn: [linkedin.com/in/seu-perfil](https://www.linkedin.com)  
- GitHub: [github.com/seu-usuario](https://github.com)
