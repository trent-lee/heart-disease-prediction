# config.py
# Central place for paths and column definitions so every script agrees on them.
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Data" / "heart.csv"
MODEL_DIR = BASE_DIR / "models"
GRAPH_DIR = BASE_DIR / "model_graphs"
DOCS_DIR = BASE_DIR / "docs"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
GUIDE_PATH = DOCS_DIR / "GUIDE.md"
GUIDE_DOCX_PATH = DOCS_DIR / "Heart_App_Setup_Guide.docx"

MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_DIR = BASE_DIR / "mlruns"
MLFLOW_EXPERIMENT = "heart-disease-prediction"
REGISTERED_MODEL_NAME = "heart-disease-classifier"

TARGET = "HeartDisease"
NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

METRIC_CHOICES = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
RANDOM_STATE = 42
