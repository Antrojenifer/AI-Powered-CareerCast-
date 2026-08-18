"""Milestone 3 – Streamlit Review UI (improved visual design)."""
from __future__ import annotations
import json, os, sys
from io import StringIO
import streamlit as st
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.predict import predict_career, get_metrics
from ml.skill_gap import build_gap_report

st.set_page_config(page_title="CareerCast M3 Review", page_icon="🎯", layout="wide")

st.markdown("""
<style>
  .main-title { font-size: 2rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem; }
  .sub { color: #64748b; margin-bottom: 1.5rem; }
  .card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 1.1rem 1.25rem; box-shadow: 0 4px 16px rgba(15,23,42,0.06);
  }
  .metric-pill {
    display: inline-block; background: #eff6ff; color: #1d4ed8;
    border-radius: 999px; padding: 0.25rem 0.75rem; font-weight: 600; font-size: 0.9rem;
  }
  .skill-chip {
    display: inline-block; margin: 0.2rem 0.3rem 0.2rem 0;
    padding: 0.25rem 0.65rem; border-radius: 999px; font-size: 0.85rem;
  }
  .chip-ok { background: #dcfce7; color: #166534; }
  .chip-miss { background: #fee2e2; color: #991b1b; }
  .priority-high { color: #b91c1c; font-weight: 700; }
  .priority-med { color: #b45309; font-weight: 700; }
  .priority-low { color: #047857; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">CareerCast · Milestone 3 Review</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Career probability · Skill gap analysis · Actionable learning plan · Report export</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Profile Input")
    name = st.text_input("Name", "Demo User")
    skills = st.text_area("Skills (comma separated)", "Python, SQL, Machine Learning, Pandas, Power BI", height=130)
    degree = st.selectbox("Degree", ["B.Tech", "B.E", "M.Tech", "MCA", "B.Sc", "M.Sc", "Not Specified"])
    experience = st.number_input("Experience (years)", 0.0, 40.0, 1.0, 0.5)
    top_k = st.slider("Top-K careers", 3, 7, 5)
    run = st.button("Run Analysis", type="primary", use_container_width=True)

if not run:
    st.info("Enter skills in the sidebar and click **Run Analysis**.")
    st.stop()

if not skills.strip():
    st.error("Please enter at least one skill.")
    st.stop()

with st.spinner("Running prediction and skill-gap analysis..."):
    result = predict_career(skills=skills, degree=degree, experience=float(experience), top_k=top_k)
    user_skills = result.get("user_skills", [])
    gap_report = build_gap_report(user_skills, result.get("top_careers", []))
    primary = gap_report.get("primary_gap") or {}

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Top Career", result["predicted_career"])
k2.metric("Confidence", f"{result['confidence']}%")
k3.metric("Skill Match", f"{primary.get('match_percent', 0)}%")
k4.metric("Model", result.get("model_used", "logistic_regression").replace("_", " ").title())

st.markdown("---")

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Career probability ranking")
    rows = [{
        "Career": i.get("career"),
        "Confidence %": i.get("confidence"),
        "Skill Match %": i.get("skill_match", i.get("alignment_score", 0)),
    } for i in result.get("top_careers", [])]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if not df.empty:
        st.bar_chart(df.set_index("Career")[["Confidence %", "Skill Match %"]])

with right:
    st.subheader("Primary skill gap")
    st.write(primary.get("summary", ""))
    st.markdown("**Matched skills**")
    matched = primary.get("matched_skills") or []
    if matched:
        chips = " ".join([f'<span class="skill-chip chip-ok">{s}</span>' for s in matched])
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.write("None")
    st.markdown("**Missing skills**")
    missing = primary.get("missing_skills") or []
    if missing:
        chips = " ".join([f'<span class="skill-chip chip-miss">{s}</span>' for s in missing])
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.write("None — strong match")

st.subheader("Actionable improvement plan")
suggestions = primary.get("suggestions") or []
if suggestions:
    for s in suggestions:
        pr = s.get("priority", "Medium")
        cls = "priority-high" if pr == "High" else ("priority-med" if pr == "Medium" else "priority-low")
        st.markdown(f"**{s['skill']}** · <span class='{cls}'>{pr}</span>  \n{s['action']}", unsafe_allow_html=True)
else:
    st.success("No major gaps detected for the primary career.")

st.subheader("Model performance")
try:
    metrics = get_metrics()
except Exception:
    metrics = {}
models = (metrics or {}).get("models", {})
if models:
    mdf = pd.DataFrame([
        {"Model": k.replace("_", " ").title(),
         "Accuracy": v.get("accuracy"),
         "Precision": v.get("precision"),
         "Recall": v.get("recall"),
         "F1": v.get("f1_score")}
        for k, v in models.items()
    ])
    st.dataframe(mdf, use_container_width=True, hide_index=True)
else:
    st.write("Metrics file not available.")

st.subheader("Export")
export_payload = {
    "name": name, "skills": skills, "degree": degree, "experience": experience,
    "prediction": result, "gap_report": gap_report,
}
c1, c2 = st.columns(2)
with c1:
    st.download_button("Download JSON report", data=json.dumps(export_payload, indent=2),
                       file_name="careercast_m3_report.json", mime="application/json", use_container_width=True)
with c2:
    buf = StringIO(); df.to_csv(buf, index=False)
    st.download_button("Download ranking CSV", data=buf.getvalue(),
                       file_name="career_ranking.csv", mime="text/csv", use_container_width=True)
