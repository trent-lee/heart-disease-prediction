# app.py — Streamlit dashboard entry point.
# Run locally:  streamlit run app.py
import streamlit as st

from dashboard import common, data_explorer, guide, performance, predict

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon=":material/cardiology:",
    layout="wide",
)
common.inject_css()
st.logo(str(common.LOGO_PATH), size="large", icon_image=str(common.ICON_PATH))

navigation = st.navigation({
    "Dashboard": [
        st.Page(predict.render, title="Risk Assessment", icon=":material/monitor_heart:",
                url_path="predict", default=True),
        st.Page(performance.render, title="Model Performance", icon=":material/insights:",
                url_path="performance"),
        st.Page(data_explorer.render, title="Data Explorer", icon=":material/table_chart:",
                url_path="data"),
    ],
    "Learn": [
        st.Page(guide.render, title="Setup & Deployment Guide", icon=":material/menu_book:",
                url_path="guide"),
    ],
})
common.sidebar_model_card()
navigation.run()
