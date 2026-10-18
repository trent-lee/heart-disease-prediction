# dashboard/data_explorer.py — explore the training dataset.
import altair as alt
import pandas as pd
import streamlit as st

from config import FEATURE_COLS, NUMERIC_COLS, TARGET
from dashboard.common import (
    CATEGORY_LABELS, FEATURE_LABELS, OUTCOME_SCALE, kpi_row, load_dataset, page_header,
    section_label,
)

CONTINUOUS_COLS = [c for c in NUMERIC_COLS if c != "FastingBS"]


def histogram(df, feature):
    return alt.Chart(df).mark_bar(opacity=0.75, binSpacing=1).encode(
        x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=30), title=FEATURE_LABELS[feature]),
        y=alt.Y("count():Q", stack=None, title="Patients"),
        color=alt.Color("Outcome:N", scale=OUTCOME_SCALE, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Outcome", alt.Tooltip("count():Q", title="Patients")],
    ).properties(height=340)


def disease_rate_chart(df, feature):
    labels = CATEGORY_LABELS.get(feature, {})
    summary = (df.groupby(feature)[TARGET].agg(["mean", "count"]).reset_index()
               .rename(columns={"mean": "Disease rate", "count": "Patients"}))
    summary["Category"] = summary[feature].map(lambda v: labels.get(v, str(v)))
    bars = alt.Chart(summary).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        x=alt.X("Disease rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Heart disease rate"),
        y=alt.Y("Category:N", sort="-x", title=None),
        color=alt.Color("Disease rate:Q", scale=alt.Scale(scheme="reds", domain=[0, 1]), legend=None),
        tooltip=["Category", alt.Tooltip("Disease rate:Q", format=".1%"), "Patients"],
    )
    text = bars.mark_text(align="left", dx=6, color="#0F172A").encode(
        text=alt.Text("Disease rate:Q", format=".0%"), color=alt.value("#0F172A"))
    return (bars + text).properties(height=max(160, 70 * len(summary)))


def correlation_chart(df):
    corr = df[NUMERIC_COLS + [TARGET]].corr().round(2)
    labels = {**FEATURE_LABELS, TARGET: "Heart disease"}
    corr.index = corr.columns = [labels[c] for c in corr.columns]
    long = corr.reset_index(names="A").melt(id_vars="A", var_name="B", value_name="r")
    base = alt.Chart(long).encode(x=alt.X("A:N", title=None, sort=list(corr.columns)),
                                  y=alt.Y("B:N", title=None, sort=list(corr.columns)))
    heat = base.mark_rect().encode(
        color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
                        legend=alt.Legend(title="Correlation")))
    text = base.mark_text(fontSize=12).encode(
        text="r:Q", color=alt.condition("abs(datum.r) > 0.5", alt.value("white"), alt.value("#0F172A")))
    return (heat + text).properties(height=430)


def render():
    df = load_dataset()
    page_header(
        "Data Explorer",
        "The UCI heart failure prediction dataset: 918 patients from five hospitals, "
        "11 clinical features and a heart disease label.",
        "Dataset",
    )

    kpi_row([
        ("Patients", f"{len(df):,}", None),
        ("Clinical features", len(FEATURE_COLS), None),
        ("Heart disease cases", f"{int(df[TARGET].sum()):,}", None),
        ("Disease rate", f"{df[TARGET].mean():.1%}", None),
    ])

    dist_tab, corr_tab, raw_tab = st.tabs(["Distributions", "Correlations", "Raw data"])

    with dist_tab:
        feature = st.selectbox("Feature", FEATURE_COLS, format_func=FEATURE_LABELS.get)
        if feature in CONTINUOUS_COLS:
            chart_col, stats_col = st.columns([2, 1], gap="medium")
            with chart_col.container(border=True):
                section_label(f"{FEATURE_LABELS[feature]} by outcome")
                st.altair_chart(histogram(df, feature), width="stretch")
            with stats_col.container(border=True):
                section_label("Summary statistics")
                stats = df.groupby("Outcome")[feature].describe()[["mean", "std", "min", "50%", "max"]]
                stats.columns = ["Mean", "Std", "Min", "Median", "Max"]
                st.dataframe(stats.T.round(1), width="stretch")
        else:
            with st.container(border=True):
                section_label(f"Heart disease rate by {FEATURE_LABELS[feature].lower()}")
                st.altair_chart(disease_rate_chart(df, feature), width="stretch")

    with corr_tab:
        with st.container(border=True):
            section_label("Correlation between numeric features and heart disease")
            st.altair_chart(correlation_chart(df), width="stretch")

    with raw_tab:
        st.dataframe(df.drop(columns="Outcome"), width="stretch", height=420)
        st.download_button("Download CSV", df.drop(columns="Outcome").to_csv(index=False),
                           file_name="heart.csv", mime="text/csv", icon=":material/download:")
