# dashboard/guide.py — renders docs/GUIDE.md as a step-by-step guide.
import re

import streamlit as st

from dashboard.common import load_guide, load_guide_docx, page_header


def split_sections(markdown):
    """Split the guide on '## ' headings, ignoring headings inside code blocks."""
    sections, title, lines, in_code = [], None, [], False
    for line in markdown.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("## "):
            if title:
                sections.append((title, "\n".join(lines).strip()))
            title, lines = line[3:].strip(), []
        elif title:
            lines.append(line)
    if title:
        sections.append((title, "\n".join(lines).strip()))
    return sections


def go_to(step):
    st.session_state["guide_step"] = step


def render():
    page_header(
        "Setup & Deployment Guide",
        "From cloning the repository to a live app on Azure: run it locally, retrain with MLflow, "
        "publish to GitHub and deploy automatically with GitHub Actions.",
        "Learn",
    )
    guide = load_guide()
    sections = split_sections(guide)
    titles = [re.sub(r"^\d+\.\s*", "", t) for t, _ in sections]

    downloads = st.container(horizontal=True)
    downloads.download_button("Download guide (Markdown)", guide, file_name="GUIDE.md",
                              mime="text/markdown", icon=":material/description:")
    docx = load_guide_docx()
    if docx:
        downloads.download_button(
            "Download guide (Word)", docx, file_name="Heart_App_Setup_Guide.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            icon=":material/article:",
        )

    st.session_state.setdefault("guide_step", 0)
    nav_col, body_col = st.columns([1, 3], gap="large")
    with nav_col.container(border=True):
        step = st.radio("Steps", range(len(sections)), key="guide_step",
                        format_func=lambda i: f"{i + 1}. {titles[i]}", label_visibility="collapsed")

    with body_col.container(border=True):
        st.caption(f"STEP {step + 1} OF {len(sections)}")
        st.subheader(titles[step], anchor=False)
        st.markdown(sections[step][1])

        prev_col, next_col = st.columns(2)
        if step > 0:
            prev_col.button(f"← {titles[step - 1]}", on_click=go_to, args=(step - 1,), width="stretch")
        if step < len(sections) - 1:
            next_col.button(f"{titles[step + 1]} →", on_click=go_to, args=(step + 1,),
                            type="primary", width="stretch")
