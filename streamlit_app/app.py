"""
Milestone 3 – Streamlit Review UI
Career probability visualization + skill gap report export.
Run: streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import json
import os
import sys
from io import StringIO

import streamlit as st
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.predict import predict_career, get_metrics
from ml.skill_gap import build_gap_report, analyze_skill_gap

st.set_page_config(
    page_title="CareerCast Review UI – Milestone 3",
    page_icon="📊",
    layout="wide",
)

st.title("CareerCast – Milestone 3 Review UI")
st.caption("Career probability visualization • Skill gap analysis • Report export")

with st.sidebar:
    st.header("Input Profile")
    name = st.text_input("Name", value="Demo User")
    skills = st.text_area(
        "Skills (comma separated)",
        value="Python, SQL, Machine Learning, Pandas, Power BI",
        height=120,
    )
    degree = st.selectbox(
        "Degree",
        ["B.Tech", "B.E", "M.Tech", "MCA", "B.Sc", "M.Sc", "Not Specified"],
    )
    experience = st.number_input("Experience (years)", min_value=0.0, max_value=40.0, value=1.0, step=0.5)
    top_k = st.slider("Top-K careers", 3, 7, 5)
    run = st.button("Run Analysis", type="primary")

if run:
    if not skills.strip():
        st.error("Please enter at least one skill.")
        st.stop()

    with st.spinner("Running prediction and gap analysis..."):
        result = predict_career(
            skills=skills,
            degree=degree,
            experience=float(experience),
            top_k=top_k,
        )
        user_skills = result.get("user_skills", [])
        gap_report = build_gap_report(user_skills, result.get("top_careers", []))

    st.success(f"Primary recommendation: **{result['predicted_career']}** ({result['confidence']}%)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Career", result["predicted_career"])
    col2.metric("Confidence", f"{result['confidence']}%")
    col3.metric("Model", result.get("model_used", "logistic_regression"))

    st.subheader("Career Probability Ranking")
    rows = []
    for item in result.get("top_careers", []):
        rows.append({
            "Career": item.get("career"),
            "Confidence %": item.get("confidence"),
            "Skill Match %": item.get("skill_match", item.get("alignment_score", 0)),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    if not df.empty:
        st.bar_chart(df.set_index("Career")["Confidence %"])

    st.subheader("Skill Gap Report")
    primary = gap_report.get("primary_gap") or {}
    st.write(primary.get("summary", ""))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matched Skills**")
        st.write(", ".join(primary.get("matched_skills") or ["None"]) )
    with c2:
        st.markdown("**Missing Skills**")
        st.write(", ".join(primary.get("missing_skills") or ["None"]))

    st.markdown("**Actionable Suggestions**")
    suggestions = primary.get("suggestions") or []
    if suggestions:
        sug_df = pd.DataFrame(suggestions)
        st.dataframe(sug_df, use_container_width=True)
    else:
        st.info("No major skill gaps detected for the primary career.")

    st.subheader("Model Metrics")
    metrics = get_metrics() if callable(globals().get("get_metrics", None)) else {}
    try:
        from ml.predict import get_metrics as gm
        metrics = gm()
    except Exception:
        metrics = {}
    if metrics:
        st.json(metrics)

    # Export
    st.subheader("Export Report")
    export_payload = {
        "name": name,
        "skills": skills,
        "degree": degree,
        "experience": experience,
        "prediction": result,
        "gap_report": gap_report,
    }
    export_text = json.dumps(export_payload, indent=2)
    st.download_button(
        "Download JSON Report",
        data=export_text,
        file_name="careercast_gap_report.json",
        mime="application/json",
    )

    # Simple CSV of ranking
    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download Ranking CSV",
        data=csv_buf.getvalue(),
        file_name="career_ranking.csv",
        mime="text/csv",
    )
else:
    st.info("Enter skills in the sidebar and click **Run Analysis**.")
