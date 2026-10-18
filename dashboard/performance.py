# dashboard/performance.py — compare trained models.
import altair as alt
import pandas as pd
import streamlit as st

from dashboard.common import (
    METRIC_LABELS, kpi_row, load_metrics, page_header, pretty_feature, require_model,
    section_label,
)

METRIC_HELP = {
    "accuracy": "Share of all test patients classified correctly.",
    "precision": "Of the patients predicted to have heart disease, the share that actually do.",
    "recall": "Of the patients who have heart disease, the share the model catches.",
    "f1_score": "Harmonic mean of precision and recall.",
    "roc_auc": "Probability the model ranks a random sick patient above a random healthy one.",
}


def comparison_chart(models):
    rows = [{"Model": name, "Metric": METRIC_LABELS[m], "Score": s[m]}
            for name, s in models.items() for m in METRIC_LABELS]
    return alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Metric:N", title=None, sort=list(METRIC_LABELS.values()),
                axis=alt.Axis(labelAngle=0)),
        xOffset="Model:N",
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.6, 1.0], clamp=True), title=None,
                axis=alt.Axis(format="%")),
        color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".3f")],
    ).properties(height=320)


def roc_chart(models):
    rows = [{"Model": f"{name} (AUC {s['roc_auc']:.3f})", "FPR": f, "TPR": t}
            for name, s in models.items()
            for f, t in zip(s["roc_curve"]["fpr"], s["roc_curve"]["tpr"])]
    curves = alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2.5, interpolate="step-after").encode(
        x=alt.X("FPR:Q", title="False positive rate"),
        y=alt.Y("TPR:Q", title="True positive rate"),
        color=alt.Color("Model:N", legend=alt.Legend(orient="bottom-right", title=None,
                                                     fillColor="white", padding=8)),
    )
    chance = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[6, 4], color="#94A3B8").encode(x="x:Q", y="y:Q")
    return (chance + curves).properties(height=320)


def confusion_chart(matrix):
    labels = ["No disease", "Heart disease"]
    rows = [{"Actual": labels[i], "Predicted": labels[j], "Patients": matrix[i][j]}
            for i in range(2) for j in range(2)]
    base = alt.Chart(pd.DataFrame(rows)).encode(
        x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("Actual:N", sort=labels),
    )
    heat = base.mark_rect(cornerRadius=6).encode(
        color=alt.Color("Patients:Q", scale=alt.Scale(scheme="blues"), legend=None))
    text = base.mark_text(fontSize=22, fontWeight="bold").encode(
        text="Patients:Q",
        color=alt.condition(alt.datum.Patients > max(map(max, matrix)) / 2,
                            alt.value("white"), alt.value("#0F172A")),
    )
    return (heat + text).properties(height=280)


def importance_chart(importance):
    df = pd.DataFrame({"Feature": [pretty_feature(k) for k in importance],
                       "Importance": list(importance.values())})
    return alt.Chart(df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                                  color="#D7263D").encode(
        x=alt.X("Importance:Q", title=None, axis=alt.Axis(format="%")),
        y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1%")],
    ).properties(height=280)


def render():
    require_model()
    metrics = load_metrics()
    models = metrics["models"]
    best = metrics["best_model"]
    page_header(
        "Model Performance",
        f"{len(models)} algorithms were tuned with 5-fold cross-validation and scored on a held-out "
        f"test set of {metrics.get('n_test', '–')} patients.",
        "Evaluation",
    )

    selection = METRIC_LABELS.get(metrics["selection_metric"], metrics["selection_metric"])
    st.markdown(f"Serving **{best}** — chosen by highest test **{selection}**. "
                f"Trained {metrics.get('trained_at', '')}.")
    kpi_row([
        (label, f"{models[best][key]:.3f}" if key == "roc_auc" else f"{models[best][key]:.1%}",
         METRIC_HELP[key])
        for key, label in METRIC_LABELS.items()
    ])

    left, right = st.columns(2, gap="medium")
    with left.container(border=True):
        section_label("Model comparison")
        st.altair_chart(comparison_chart(models), width="stretch")
    with right.container(border=True):
        section_label("ROC curves")
        st.altair_chart(roc_chart(models), width="stretch")

    chosen = st.segmented_control("Inspect a model", list(models), default=best, required=True)
    left, right = st.columns(2, gap="medium")
    with left.container(border=True):
        section_label(f"Confusion matrix — {chosen}")
        st.altair_chart(confusion_chart(models[chosen]["confusion_matrix"]), width="stretch")
    with right.container(border=True):
        section_label(f"Feature importance — {chosen}")
        if models[chosen].get("feature_importance"):
            st.altair_chart(importance_chart(models[chosen]["feature_importance"]), width="stretch")
        else:
            st.info("This model does not expose feature importances.")

    with st.expander("Tuned hyper-parameters and cross-validation scores"):
        table = pd.DataFrame([
            {"Model": name, "CV accuracy": f"{s['cv_accuracy']:.3f}",
             "Best parameters": ", ".join(f"{k}={v}" for k, v in s["best_params"].items())}
            for name, s in models.items()
        ])
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Features kept by feature selection: "
                   + ", ".join(pretty_feature(f) for f in metrics["selected_features"]))
