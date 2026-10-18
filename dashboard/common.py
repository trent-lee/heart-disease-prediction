# dashboard/common.py — shared loaders, labels, styling and layout helpers.
import json

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from config import (
    BASE_DIR, BEST_MODEL_PATH, CATEGORICAL_COLS, DATA_PATH, GUIDE_DOCX_PATH, GUIDE_PATH,
    METRICS_PATH, TARGET,
)

LOGO_PATH = BASE_DIR / "assets" / "logo.svg"

OUTCOME_LABELS = {0: "No disease", 1: "Heart disease"}
OUTCOME_SCALE = alt.Scale(domain=["No disease", "Heart disease"], range=["#2563EB", "#D7263D"])

FEATURE_LABELS = {
    "Age": "Age",
    "Sex": "Sex",
    "ChestPainType": "Chest pain type",
    "RestingBP": "Resting blood pressure",
    "Cholesterol": "Cholesterol",
    "FastingBS": "Fasting blood sugar > 120",
    "RestingECG": "Resting ECG",
    "MaxHR": "Max heart rate",
    "ExerciseAngina": "Exercise angina",
    "Oldpeak": "Oldpeak (ST depression)",
    "ST_Slope": "ST slope",
}
CATEGORY_LABELS = {
    "Sex": {"M": "Male", "F": "Female"},
    "ChestPainType": {
        "ASY": "Asymptomatic",
        "ATA": "Atypical angina",
        "NAP": "Non-anginal pain",
        "TA": "Typical angina",
    },
    "RestingECG": {"Normal": "Normal", "ST": "ST-T abnormality", "LVH": "LV hypertrophy"},
    "ExerciseAngina": {"N": "No", "Y": "Yes"},
    "ST_Slope": {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"},
    "FastingBS": {0: "No", 1: "Yes"},
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 score",
    "roc_auc": "ROC-AUC",
}

CSS = """
<style>
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1280px; }
.hero {
    background: linear-gradient(120deg, #0F172A 0%, #1E293B 55%, #7F1D1D 100%);
    border-radius: 16px; padding: 1.6rem 2rem; margin-bottom: 1.4rem; color: #F8FAFC;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
}
.hero .eyebrow {
    text-transform: uppercase; letter-spacing: .14em; font-size: .72rem;
    font-weight: 700; color: #FCA5A5; margin-bottom: .35rem;
}
.hero .title { font-size: 1.9rem; font-weight: 750; line-height: 1.2; color: #FFFFFF; }
.hero .subtitle { margin-top: .45rem; color: #CBD5E1; font-size: 1rem; max-width: 880px; }
.section-label {
    text-transform: uppercase; letter-spacing: .08em; font-size: .74rem; font-weight: 700;
    color: #64748B; margin: .4rem 0 .2rem 0;
}
.risk-badge {
    display: inline-block; padding: .35rem .9rem; border-radius: 999px;
    font-weight: 700; font-size: .95rem;
}
[class*="st-key-kpi-"] { min-width: 150px; box-sizing: border-box; }
.empty-state { text-align: center; padding: 2.4rem 1rem; color: #64748B; }
.empty-state .icon { font-size: 2.6rem; }
.sidebar-card {
    background: #1E293B; border: 1px solid #334155; border-radius: 12px;
    padding: .9rem 1rem; font-size: .85rem; color: #CBD5E1; line-height: 1.6;
}
.sidebar-card b { color: #F8FAFC; }
.sidebar-card .muted { color: #94A3B8; font-size: .78rem; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title, subtitle, eyebrow):
    st.markdown(
        f"""<div class="hero"><div class="eyebrow">{eyebrow}</div>
        <div class="title">{title}</div><div class="subtitle">{subtitle}</div></div>""",
        unsafe_allow_html=True,
    )


def section_label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def kpi_row(items):
    """Bordered metric cards that wrap onto a new line on narrow screens."""
    row = st.container(horizontal=True, gap="small")
    for label, value, help_text in items:
        key = "kpi-" + "".join(c if c.isalnum() else "-" for c in label.lower())
        with row.container(border=True, width="stretch", key=key):
            st.metric(label, value, help=help_text)


def pretty_feature(name):
    """'ST_Slope_Up' -> 'ST slope: Upsloping', 'MaxHR' -> 'Max heart rate'."""
    for col in CATEGORICAL_COLS:
        if name.startswith(f"{col}_"):
            value = name[len(col) + 1:]
            return f"{FEATURE_LABELS[col]}: {CATEGORY_LABELS[col].get(value, value)}"
    return FEATURE_LABELS.get(name, name)


def model_ready():
    return BEST_MODEL_PATH.exists() and METRICS_PATH.exists()


def require_model():
    if not model_ready():
        st.error("No trained model found. Run `python main.py` in the project folder first, "
                 "then refresh this page.", icon=":material/error:")
        st.stop()


@st.cache_resource
def load_model():
    return joblib.load(BEST_MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df["Outcome"] = df[TARGET].map(OUTCOME_LABELS)
    return df


@st.cache_data
def load_guide():
    return GUIDE_PATH.read_text(encoding="utf-8")


@st.cache_data
def load_guide_docx():
    return GUIDE_DOCX_PATH.read_bytes() if GUIDE_DOCX_PATH.exists() else None


def sidebar_model_card():
    if not model_ready():
        return
    metrics = load_metrics()
    best = metrics["best_model"]
    scores = metrics["models"][best]
    registry = metrics.get("registered_model", {})
    st.sidebar.markdown(
        f"""<div class="sidebar-card">
        <div class="muted">MODEL IN PRODUCTION</div>
        <b>{best}</b><br>
        Accuracy <b>{scores['accuracy']:.1%}</b> · ROC-AUC <b>{scores['roc_auc']:.3f}</b><br>
        <span class="muted">MLflow version {registry.get('version', '–')} ·
        trained {metrics.get('trained_at', '')}</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Educational project — not a medical diagnosis tool.")
