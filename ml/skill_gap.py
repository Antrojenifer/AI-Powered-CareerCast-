"""
Milestone 3 – Skill Gap Analysis Module
Compares user skills with target career requirements and returns
actionable competency improvement suggestions.
"""
from __future__ import annotations

from typing import Dict, List, Any

# Target skill profiles per career (normalized lowercase keys matched against user skills)
CAREER_SKILLS: Dict[str, List[str]] = {
    "Data Scientist": [
        "Python", "Machine Learning", "Pandas", "Numpy", "Scikit-Learn",
        "Statistics", "Sql", "Deep Learning", "Data Analysis",
    ],
    "Data Analyst": [
        "Sql", "Power Bi", "Excel", "Tableau", "Python",
        "Data Analysis", "Statistics", "Pandas",
    ],
    "ML Engineer": [
        "Python", "Deep Learning", "Tensorflow", "Pytorch", "Machine Learning",
        "Scikit-Learn", "Mlops", "Docker", "Aws",
    ],
    "AI Engineer": [
        "Python", "Gen Ai", "Llm", "Langchain", "Deep Learning",
        "Transformers", "Huggingface", "Machine Learning",
    ],
    "Backend Developer": [
        "Python", "Java", "Sql", "Flask", "Django", "Rest Api", "Docker",
    ],
    "Frontend Developer": [
        "Html", "Css", "Javascript", "React", "Typescript",
    ],
    "Full Stack Developer": [
        "Html", "Css", "Javascript", "React", "Node.Js", "Python", "Sql",
    ],
    "DevOps Engineer": [
        "Aws", "Docker", "Kubernetes", "Ci/Cd", "Linux", "Devops", "Terraform",
    ],
    "Data Engineer": [
        "Sql", "Python", "Spark", "Hadoop", "Etl", "Data Engineering", "Airflow",
    ],
    "NLP Engineer": [
        "Python", "Nlp", "Transformers", "Spacy", "Bert", "Huggingface",
    ],
    "Cybersecurity Analyst": [
        "Cybersecurity", "Linux", "Networking", "Python",
    ],
    "UI/UX Designer": ["Figma", "Ui/Ux", "Html", "Css"],
    "Cloud Engineer": ["Aws", "Azure", "Gcp", "Docker", "Kubernetes", "Linux"],
    "Software Engineer": [
        "Python", "Java", "Javascript", "Sql", "Git", "Data Structures",
    ],
}

# Simple learning suggestions for missing skills
SKILL_ACTIONS: Dict[str, str] = {
    "python": "Complete a Python fundamentals course and practice coding problems weekly.",
    "sql": "Practice SQL joins, aggregations, and window functions on sample datasets.",
    "machine learning": "Study supervised learning basics and train small models with scikit-learn.",
    "deep learning": "Learn neural network basics using TensorFlow or PyTorch tutorials.",
    "pandas": "Practice data cleaning and analysis with Pandas on public CSV datasets.",
    "numpy": "Review NumPy arrays, broadcasting, and basic numerical operations.",
    "scikit-learn": "Build classification and regression pipelines using scikit-learn.",
    "tensorflow": "Follow official TensorFlow beginner tutorials for model training.",
    "pytorch": "Complete a PyTorch intro course covering tensors and training loops.",
    "power bi": "Build 2–3 dashboard projects in Power BI using public data.",
    "tableau": "Create interactive dashboards in Tableau Public.",
    "excel": "Learn pivot tables, VLOOKUP/XLOOKUP, and basic data analysis in Excel.",
    "html": "Build static web pages and practice semantic HTML.",
    "css": "Practice responsive layouts with Flexbox and Grid.",
    "javascript": "Learn DOM basics and build small interactive web pages.",
    "react": "Build a small multi-component React app with hooks.",
    "node.js": "Create a simple REST API using Node.js and Express.",
    "docker": "Containerize a sample app and run it with Docker Compose.",
    "kubernetes": "Complete a beginner Kubernetes lab (pods, deployments, services).",
    "aws": "Complete AWS Cloud Practitioner fundamentals and try core services.",
    "azure": "Explore Azure fundamentals and deploy a simple web app.",
    "gcp": "Try Google Cloud free-tier basics (Compute, Storage).",
    "flask": "Build a small Flask API with 2–3 endpoints.",
    "django": "Build a basic Django CRUD application.",
    "rest api": "Design and document a sample REST API with request/response examples.",
    "gen ai": "Experiment with prompt engineering and LLM API usage.",
    "llm": "Learn how large language models work at a high level and try API calls.",
    "langchain": "Build a simple RAG or agent demo with LangChain.",
    "transformers": "Use Hugging Face Transformers for a text classification demo.",
    "huggingface": "Deploy or run a pretrained model from the Hugging Face hub.",
    "nlp": "Practice text preprocessing, tokenization, and basic NLP tasks.",
    "spacy": "Use spaCy for named entity recognition on sample documents.",
    "spark": "Run basic Spark DataFrame transformations on sample data.",
    "etl": "Design a simple extract-transform-load pipeline for CSV data.",
    "ci/cd": "Set up a basic GitHub Actions workflow for tests.",
    "linux": "Practice common Linux shell commands and file permissions.",
    "git": "Practice branching, commits, pull requests, and merges.",
    "statistics": "Revise probability, distributions, and hypothesis testing basics.",
    "data analysis": "Complete an end-to-end analysis project and write findings.",
    "figma": "Design a simple mobile or web UI prototype in Figma.",
}


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _normalize_skill_list(skills: List[str]) -> List[str]:
    out = []
    seen = set()
    for s in skills or []:
        n = _norm(str(s))
        if n and n not in seen:
            seen.add(n)
            out.append(str(s).strip())
    return out


def analyze_skill_gap(
    user_skills: List[str],
    target_career: str,
    career_skills_map: Dict[str, List[str]] | None = None,
) -> Dict[str, Any]:
    """
    Compare user skills with target career skills.
    Returns matched, missing, match percent, and actionable suggestions.
    """
    cmap = career_skills_map or CAREER_SKILLS
    required = cmap.get(target_career, [])
    if not required:
        # fuzzy key match
        key = None
        tc = _norm(target_career)
        for k in cmap:
            if tc in _norm(k) or _norm(k) in tc:
                key = k
                break
        required = cmap.get(key or "", [])

    user_list = _normalize_skill_list(user_skills)
    user_norm = {_norm(s) for s in user_list}
    req_norm = [(_norm(s), s) for s in required]

    matched = []
    missing = []
    for rn, original in req_norm:
        if rn in user_norm or any(rn in u or u in rn for u in user_norm):
            matched.append(original)
        else:
            missing.append(original)

    total = len(required) if required else 1
    match_pct = round(100.0 * len(matched) / total, 1)

    suggestions = []
    for skill in missing[:8]:
        action = SKILL_ACTIONS.get(_norm(skill), f"Learn the fundamentals of {skill} and complete a small hands-on project.")
        suggestions.append({
            "skill": skill,
            "priority": "High" if match_pct < 50 else ("Medium" if match_pct < 75 else "Low"),
            "action": action,
        })

    return {
        "target_career": target_career,
        "required_skills": required,
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percent": match_pct,
        "alignment_score": match_pct,
        "suggestions": suggestions,
        "summary": (
            f"You match {match_pct}% of skills for {target_career}. "
            f"{len(matched)} matched, {len(missing)} missing."
        ),
    }


def build_gap_report(
    user_skills: List[str],
    top_careers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a multi-career skill gap report from ranked career predictions.
    top_careers items should include at least: career, confidence
    """
    reports = []
    for item in top_careers:
        career = item.get("career") or item.get("predicted_career") or ""
        gap = analyze_skill_gap(user_skills, career)
        reports.append({
            "career": career,
            "confidence": item.get("confidence", 0),
            "match_percent": gap["match_percent"],
            "matched_skills": gap["matched_skills"],
            "missing_skills": gap["missing_skills"],
            "suggestions": gap["suggestions"],
            "summary": gap["summary"],
        })

    primary = reports[0] if reports else None
    return {
        "primary_career": primary["career"] if primary else None,
        "primary_gap": primary,
        "career_gaps": reports,
        "user_skills": _normalize_skill_list(user_skills),
    }
