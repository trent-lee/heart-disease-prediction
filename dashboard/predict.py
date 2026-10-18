# dashboard/predict.py — patient form and risk estimate.
import altair as alt
import pandas as pd
import streamlit as st

from config import FEATURE_COLS
from dashboard.common import (
    CATEGORY_LABELS, FEATURE_LABELS, load_dataset, load_model, page_header, require_model,
    section_label,
)

RISK_BANDS = [
    (0.35, "Low risk", "#16A34A", "#DCFCE7",
     "The model sees a low likelihood of heart disease for this profile."),
    (0.65, "Moderate risk", "#D97706", "#FEF3C7",
     "Some indicators are associated with heart disease. A clinical follow-up is advisable."),
    (1.01, "High risk", "#DC2626", "#FEE2E2",
     "Several strong indicators of heart disease are present. Refer for cardiology review."),
]
COMPARE_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]


def risk_band(probability):
    for upper, label, color, background, message in RISK_BANDS:
        if probability < upper:
            return label, color, background, message


def risk_gauge(probability, color):
    arc_data = pd.DataFrame({"part": ["risk", "rest"], "value": [probability, 1 - probability],
                             "order": [0, 1]})
    arc = alt.Chart(arc_data).mark_arc(innerRadius=78, outerRadius=104, cornerRadius=6).encode(
        theta=alt.Theta("value:Q", stack=True),
        order=alt.Order("order:Q"),
        color=alt.Color("part:N", scale=alt.Scale(domain=["risk", "rest"], range=[color, "#E2E8F0"]),
                        legend=None),
        tooltip=alt.value(None),
    )
    value = alt.Chart(pd.DataFrame({"t": [f"{probability:.0%}"]})).mark_text(
        fontSize=38, fontWeight="bold", color="#0F172A", dy=-6).encode(text="t:N")
    label = alt.Chart(pd.DataFrame({"t": ["probability"]})).mark_text(
        fontSize=13, color="#64748B", dy=24).encode(text="t:N")
    return alt.layer(arc, value, label).properties(height=230)


def patient_form():
    with st.form("patient_form", border=True):
        section_label("Demographics")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", min_value=18, max_value=100, value=54)
        sex = c2.selectbox("Sex", ["M", "F"], format_func=CATEGORY_LABELS["Sex"].get)

        section_label("Vitals & lab results")
        c1, c2 = st.columns(2)
        resting_bp = c1.number_input("Resting blood pressure (mm Hg)", min_value=0, max_value=250,
                                     value=130)
        cholesterol = c2.number_input("Serum cholesterol (mg/dl)", min_value=0, max_value=700,
                                      value=220, help="Use 0 if the value was not measured.")
        c1, c2 = st.columns(2)
        max_hr = c1.number_input("Max heart rate achieved (bpm)", min_value=50, max_value=250,
                                 value=140)
        fasting_bs = c2.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                                  format_func=CATEGORY_LABELS["FastingBS"].get)

        section_label("Clinical findings")
        c1, c2 = st.columns(2)
        chest_pain = c1.selectbox("Chest pain type", list(CATEGORY_LABELS["ChestPainType"]),
                                  format_func=CATEGORY_LABELS["ChestPainType"].get)
        resting_ecg = c2.selectbox("Resting ECG", list(CATEGORY_LABELS["RestingECG"]),
                                   format_func=CATEGORY_LABELS["RestingECG"].get)
        c1, c2, c3 = st.columns(3)
        exercise_angina = c1.selectbox("Exercise-induced angina", ["N", "Y"],
                                       format_func=CATEGORY_LABELS["ExerciseAngina"].get)
        st_slope = c2.selectbox("ST slope", list(CATEGORY_LABELS["ST_Slope"]),
                                format_func=CATEGORY_LABELS["ST_Slope"].get)
        oldpeak = c3.number_input("Oldpeak (ST depression)", min_value=-3.0, max_value=7.0,
                                  value=1.0, step=0.1)

        submitted = st.form_submit_button("Assess risk", type="primary", width="stretch",
                                          icon=":material/monitor_heart:")

    if submitted:
        st.session_state["patient"] = {
            "Age": age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp,
            "Cholesterol": cholesterol, "FastingBS": fasting_bs, "RestingECG": resting_ecg,
            "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak,
            "ST_Slope": st_slope,
        }


def result_panel(patient):
    if patient is None:
        with st.container(border=True):
            st.markdown(
                """<div class="empty-state"><div class="icon">🩺</div>
                <b>No assessment yet</b><br>Fill in the patient details and select
                <b>Assess risk</b> to see the model's estimate.</div>""",
                unsafe_allow_html=True,
            )
        return

    model = load_model()
    probability = float(model.predict_proba(pd.DataFrame([patient])[FEATURE_COLS])[0, 1])
    label, color, background, message = risk_band(probability)

    with st.container(border=True):
        section_label("Estimated risk of heart disease")
        st.altair_chart(risk_gauge(probability, color), width="stretch")
        st.markdown(
            f'<div style="text-align:center"><span class="risk-badge" '
            f'style="color:{color};background:{background}">{label}</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(message)

    df = load_dataset()
    means = df.groupby("Outcome")[COMPARE_COLS].mean()
    comparison = pd.DataFrame({
        "Measure": [FEATURE_LABELS[c] for c in COMPARE_COLS],
        "Patient": [patient[c] for c in COMPARE_COLS],
        "Healthy avg.": means.loc["No disease"].round(1).values,
        "Disease avg.": means.loc["Heart disease"].round(1).values,
    })
    with st.container(border=True):
        section_label("How this patient compares")
        st.dataframe(comparison, hide_index=True, width="stretch")


def render():
    require_model()
    page_header(
        "Heart Disease Risk Assessment",
        "Enter a patient's clinical measurements to estimate the probability of heart disease "
        "using the best model from the latest training run.",
        "Prediction",
    )
    form_col, result_col = st.columns([3, 2], gap="large")
    with form_col:
        patient_form()
    with result_col:
        result_panel(st.session_state.get("patient"))
    st.caption("Risk bands: low < 35% ≤ moderate < 65% ≤ high. The model classifies a patient as "
               "positive at 50%. For education only — not a substitute for clinical judgement.")
