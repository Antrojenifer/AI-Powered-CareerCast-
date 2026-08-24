"""
CareerCast – Milestone 1+2+3 Streamlit demo
- Skills paste OR resume upload
- Career ranking
- Skills you have vs skills to learn (per domain)
- Actionable improvement plan + export
"""
from __future__ import annotations

import json
import os
import sys
from io import StringIO

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.predict import predict_career, get_metrics
from ml.skill_gap import analyze_skill_gap, build_gap_report

try:
    from ml.skill_gap import build_learning_roadmap
except Exception:
    build_learning_roadmap = None

st.set_page_config(page_title="CareerCast · All Milestones", page_icon="🎯", layout="wide")

st.markdown(
    """
<style>
  .chip {display:inline-block;padding:0.28rem 0.7rem;border-radius:999px;margin:0.15rem 0.25rem 0.15rem 0;font-size:0.82rem;font-weight:600;}
  .ok {background:#dcfce7;color:#166534;}
  .miss {background:#fee2e2;color:#991b1b;}
  .phase {background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:0.75rem 0.9rem;margin-bottom:0.55rem;}
  .badge {background:#dbeafe;color:#1d4ed8;font-size:0.72rem;font-weight:700;padding:0.15rem 0.45rem;border-radius:999px;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("AI-Powered Career Intelligence Platform")
st.caption("Milestone 1 · 2 · 3 — prediction, ranking, skill gap, learning plan (one app)")


def extract_resume_text(uploaded) -> str:
    """Extract text from PDF/DOCX if libraries exist; else return empty."""
    if uploaded is None:
        return ""
    name = (uploaded.name or "").lower()
    data = uploaded.read()
    # PDF
    if name.endswith(".pdf"):
        try:
            import pdfplumber
            from io import BytesIO
            text_parts = []
            with pdfplumber.open(BytesIO(data)) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception:
            st.warning("PDF parser not available on this host. Please paste skills manually.")
            return ""
    # DOCX
    if name.endswith(".docx"):
        try:
            import docx
            from io import BytesIO
            document = docx.Document(BytesIO(data))
            return "\n".join(p.text for p in document.paragraphs)
        except Exception:
            st.warning("DOCX parser not available on this host. Please paste skills manually.")
            return ""
    st.warning("Please upload PDF or DOCX, or paste skills.")
    return ""


def simple_skill_guess(text: str) -> str:
    """Very light keyword pull if full NLP model is unavailable."""
    catalog = [
        "Python", "Java", "JavaScript", "TypeScript", "SQL", "HTML", "CSS", "React", "Node.js",
        "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Linux", "Git", "Flask", "Django",
        "Machine Learning", "Deep Learning", "Pandas", "NumPy", "Power BI", "Tableau",
        "TensorFlow", "PyTorch", "Scikit-Learn", "CI/CD", "Terraform", "MongoDB", "Spring Boot",
        "Microservices", "GraphQL", "Agile",
    ]
    low = text.lower()
    found = []
    for s in catalog:
        if s.lower() in low:
            found.append(s)
    return ", ".join(found)


with st.sidebar:
    st.header("Profile input")
    name = st.text_input("Name", "Demo User")
    input_mode = st.radio("Input method", ["Paste skills", "Upload resume"], horizontal=True)

    skills_text = ""
    if input_mode == "Paste skills":
        skills_text = st.text_area(
            "Your skills (comma separated)",
            "Python, SQL, Machine Learning, Pandas, Power BI",
            height=120,
        )
    else:
        uploaded = st.file_uploader("Upload resume (PDF/DOCX)", type=["pdf", "docx"])
        if uploaded is not None:
            raw = extract_resume_text(uploaded)
            if raw.strip():
                guessed = simple_skill_guess(raw)
                st.success("Resume text read. Skills auto-detected below (you can edit).")
                skills_text = st.text_area("Extracted / editable skills", guessed or raw[:500], height=120)
            else:
                skills_text = st.text_area(
                    "Paste skills from resume",
                    "Python, SQL, Docker",
                    height=120,
                    help="If auto-extract fails, paste skills here.",
                )
        else:
            st.info("Upload a PDF/DOCX resume, or switch to Paste skills.")

    degree = st.selectbox("Degree", ["B.Tech", "B.E", "M.Tech", "MCA", "B.Sc", "Not Specified"])
    experience = st.number_input("Experience (years)", 0.0, 40.0, 1.0, 0.5)
    top_k = st.slider("Top-K careers", 3, 7, 5)
    run = st.button("Run Analysis", type="primary", use_container_width=True)

if not run:
    st.info("Choose **Paste skills** or **Upload resume**, then click **Run Analysis**.")
    st.stop()

if not (skills_text or "").strip():
    st.error("Please provide skills (paste or extract from resume).")
    st.stop()

with st.spinner("Running prediction, ranking, and skill-gap analysis..."):
    result = predict_career(
        skills=skills_text,
        degree=degree,
        experience=float(experience),
        top_k=top_k,
    )
    user_skills = result.get("user_skills", [])
    top_careers = result.get("top_careers", [])
    gap_report = build_gap_report(user_skills, top_careers)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Top Career", result.get("predicted_career", "-"))
c2.metric("Confidence", f"{result.get('confidence', 0)}%")
primary_gap = gap_report.get("primary_gap") or analyze_skill_gap(user_skills, result.get("predicted_career", ""))
c3.metric("Skill Match", f"{primary_gap.get('match_percent', primary_gap.get('match_percent', 0))}%")
c4.metric("Model", str(result.get("model_used", "logistic_regression")).replace("_", " ").title())

st.subheader("Career ranking")
rows = []
for i, item in enumerate(top_careers, 1):
    rows.append(
        {
            "Rank": i,
            "Career": item.get("career"),
            "Confidence %": item.get("confidence"),
            "Skill Match %": item.get("skill_match", item.get("alignment_score", 0)),
        }
    )
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)
if not df.empty:
    st.bar_chart(df.set_index("Career")[["Confidence %", "Skill Match %"]])

st.subheader("Select a recommended domain")
st.caption("Example: choose DevOps / Full Stack / Data Scientist to see skills you already have vs skills to learn.")
career_names = [r["Career"] for r in rows]
selected = st.selectbox("Domain", career_names, index=0)

gap = analyze_skill_gap(user_skills, selected)
matched = gap.get("matched_skills") or gap.get("matched") or []
missing = gap.get("missing_skills") or gap.get("missing") or []

left, right = st.columns(2)
with left:
    st.markdown(f"### Skills you already have for **{selected}**")
    st.caption("From your resume / entered skills that match this domain.")
    if matched:
        st.markdown(
            " ".join([f'<span class="chip ok">{s}</span>' for s in matched]),
            unsafe_allow_html=True,
        )
    else:
        st.write("No strong overlapping skills listed.")

with right:
    st.markdown(f"### Skills you need to learn for **{selected}**")
    st.caption("Required skills to improve fit for this domain.")
    if missing:
        st.markdown(
            " ".join([f'<span class="chip miss">{s}</span>' for s in missing]),
            unsafe_allow_html=True,
        )
    else:
        st.success("Strong match — no major gaps.")

st.subheader("Actionable improvement plan")
if roadmap and roadmap.get("phases"):
    st.caption(roadmap.get("summary", ""))
    for ph in roadmap["phases"]:
        items = ph.get("items") or []
        st.markdown(f"#### Phase {ph.get('phase')}: {ph.get('title')} ({ph.get('duration')})")
        st.caption(ph.get("focus", ""))
        if items:
            for it in items:
                skill = it.get("skill", "")
                action = it.get("action", "")
                st.markdown(f"- **{skill}** — {action}")
        else:
            st.markdown("- Practice core skills and complete a small project for this phase.")
        st.write("")
else:
    if missing:
        for s in missing:
            st.markdown(f"**{s}** — Learn fundamentals of {s} and complete a small hands-on task.")
    else:
        st.write("Maintain current skills and build a portfolio project for this domain.")

st.subheader("Export")
payload = {
    "name": name,
    "skills_input": skills_text,
    "selected_domain": selected,
    "skills_already_have": matched,
    "skills_to_learn": missing,
    "prediction": result,
}
st.download_button(
    "Download JSON report",
    data=json.dumps(payload, indent=2),
    file_name="careercast_report.json",
    mime="application/json",
)
csv_buf = StringIO()
df.to_csv(csv_buf, index=False)
st.download_button("Download ranking CSV", data=csv_buf.getvalue(), file_name="ranking.csv", mime="text/csv")
