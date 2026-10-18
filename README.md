# Heart Disease Prediction System

An end-to-end machine-learning project: train heart disease classifiers locally, track experiments with **MLflow**, explore results in a **Streamlit** dashboard, and deploy automatically to a free **Azure App Service** with **GitHub Actions**.

**Full step-by-step guide:** [docs/GUIDE.md](docs/GUIDE.md) (also available as a [Word document](docs/Heart_App_Setup_Guide.docx) and inside the dashboard under *Setup & Deployment Guide*).

## Quick start

```bash
git clone https://github.com/Abhi1727/heart-disease-azure-starter.git
cd heart-disease-azure-starter

python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

streamlit run app.py                 # dashboard  → http://localhost:8501
python main.py                       # retrain    → logs to MLflow, updates models/
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # MLflow UI → http://localhost:5001
```

## Dashboard

| Page | What it shows |
|---|---|
| Risk Assessment | Patient form → probability of heart disease, risk band, comparison with dataset averages |
| Model Performance | Metrics of the served model, model comparison, ROC curves, confusion matrix, feature importance |
| Data Explorer | Feature distributions by outcome, disease rate by category, correlations, raw data |
| Setup & Deployment Guide | This project's guide, step by step |

## Results (latest training run)

XGBoost — accuracy **87.0%**, ROC-AUC **0.913** on a held-out test set of 184 patients.

*Educational project — not a medical diagnosis tool.*
