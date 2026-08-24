"""
CareerCast – Milestone 1+2+3 Streamlit demo
Domain dropdown updates skills have/need WITHOUT re-running full analysis.
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

from ml.predict import predict_career
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
</style>
""",
    unsafe_allow_html=True,
)

st.title("AI-Powered Career Intelligence Platform")
st.caption("Milestone 1 · 2 · 3 — prediction, ranking, skill gap, learning plan")


def extract_resume_text(uploaded) -> str:
    if uploaded is None:
        return ""
    name = (uploaded.name or "").lower()
    data = uploaded.read()
    if name.endswith(".pdf"):
        try:
            import pdfplumber
            from io import BytesIO
            parts = []
            with pdfplumber.open(BytesIO(data)) as pdf:
                for page in pdf.pages:
                    parts.append(page.extract_text() or "")
            return "\n".join(parts)
        except Exception:
            st.warning("PDF parser not available. Please paste skills manually.")
            return ""
    if name.endswith(".docx"):
        try:
            import docx
            from io import BytesIO
            document = docx.Document(BytesIO(data))
            return "\n".join(p.text for p in document.paragraphs)
        except Exception:
            st.warning("DOCX parser not available. Please paste skills manually.")
            return ""
    return ""


def simple_skill_guess(text: str) -> str:
    catalog = [
        "Python", "Java", "JavaScript", "TypeScript", "SQL", "HTML", "CSS", "React", "Node.js",
        "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Linux", "Git", "Flask", "Django",
        "Machine Learning", "Deep Learning", "Pandas", "NumPy", "Power BI", "Tableau",
        "TensorFlow", "PyTorch", "Scikit-Learn", "CI/CD", "Terraform", "MongoDB", "Spring Boot",
        "Microservices", "GraphQL", "Agile",
    ]
    low = text.lower()
    found = [s for s in catalog if s.lower() in low]
    return ", ".join(found)


def make_plan(user_skills, selected, missing):
    if build_learning_roadmap:
        try:
            return build_learning_roadmap(user_skills, selected)
        except Exception:
            pass
    ms = missing or []
    return {
        "summary": f"Personalized plan for {selected} ({len(ms)} skills to improve).",
        "phases": [
            {
                "phase": 1,
                "title": "Foundation",
                "duration": "2–4 weeks",
                "focus": "Close critical skill gaps",
                "items": [
                    {"skill": s, "action": f"Learn the fundamentals of {s} and practice with small exercises."}
                    for s in ms[:2]
                ]
                or [{"skill": "Strengthen foundations", "action": f"Revise your current skills for {selected}."}],
            },
            {
                "phase": 2,
                "title": "Build & Practice",
                "duration": "4–6 weeks",
                "focus": "Build applied competence",
                "items": [
                    {"skill": s, "action": f"Apply {s} in a mini project."}
                    for s in ms[2:5]
                ]
                or [{"skill": "Hands-on project", "action": f"Build 1–2 projects related to {selected}."}],
            },
            {
                "phase": 3,
                "title": "Portfolio & Jobs",
                "duration": "4–8 weeks",
                "focus": "Prove readiness for roles",
                "items": [
                    {"skill": "Portfolio", "action": f"Publish projects and align your resume toward {selected} roles."}
                ],
            },
        ],
    }


with st.sidebar:
    st.header("Profile input")
    name = st.text_input("Name", "Demo User")
    input_mode = st.radio("Input method", ["Paste skills", "Upload resume"], horizontal=True)

    skills_text = ""
    if input_mode == "Paste skills":
        skills_text = st.text_area(
            "Your skills (comma separated)",
            "Python, SQL, Machine Learning, Pandas, Power BI, HTML, CSS, JavaScript",
            height=120,
        )
    else:
        uploaded = st.file_uploader("Upload resume (PDF/DOCX)", type=["pdf", "docx"])
        if uploaded is not None:
            raw = extract_resume_text(uploaded)
            if raw.strip():
                guessed = simple_skill_guess(raw)
                st.success("Resume read. Edit skills if needed.")
                skills_text = st.text_area("Extracted / editable skills", guessed or raw[:500], height=120)
            else:
                skills_text = st.text_area("Paste skills from resume", "Python, SQL, Docker", height=120)
        else:
            st.info("Upload PDF/DOCX or switch to Paste skills.")

    degree = st.selectbox("Degree", ["B.Tech", "B.E", "M.Tech", "MCA", "B.Sc", "Not Specified"])
    experience = st.number_input("Experience (years)", 0.0, 40.0, 1.0, 0.5)
    top_k = st.slider("Top-K careers", 3, 7, 5)
    run = st.button("Run Analysis", type="primary", use_container_width=True)

# Run analysis only when button clicked; store in session_state
if run:
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
        st.session_state["result"] = result
        st.session_state["user_skills"] = result.get("user_skills", [])
        st.session_state["name"] = name
        st.session_state["skills_text"] = skills_text

if "result" not in st.session_state:
    st.info("Choose **Paste skills** or **Upload resume**, then click **Run Analysis**.")
    st.stop()

result = st.session_state["result"]
user_skills = st.session_state.get("user_skills", [])
top_careers = result.get("top_careers", [])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Top Career", result.get("predicted_career", "-"))
c2.metric("Confidence", f"{result.get('confidence', 0)}%")
primary_gap = analyze_skill_gap(user_skills, result.get("predicted_career", ""))
c3.metric("Skill Match", f"{primary_gap.get('match_percent', 0)}%")
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
st.caption("Change the domain below — skills you have and skills to learn update immediately (no full page restart of analysis).")

career_names = [r["Career"] for r in rows] or [result.get("predicted_career", "Software Engineer")]
selected = st.selectbox("Domain", career_names, key="domain_select")

# Always recompute gap for the currently selected domain
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
roadmap = make_plan(user_skills, selected, missing)
st.caption(roadmap.get("summary", ""))
for ph in roadmap.get("phases", []):
    items = ph.get("items") or []
    st.markdown(f"#### Phase {ph.get('phase')}: {ph.get('title')} ({ph.get('duration')})")
    st.caption(ph.get("focus", ""))
    if items:
        for it in items:
            st.markdown(f"- **{it.get('skill', '')}** — {it.get('action', '')}")
    else:
        st.markdown("- Practice core skills and complete a small project for this phase.")

st.subheader("Export")
payload = {
    "name": st.session_state.get("name", name),
    "skills_input": st.session_state.get("skills_text", skills_text),
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
