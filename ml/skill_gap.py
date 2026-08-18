"""Milestone 3 – Skill Gap Analysis with actionable suggestions."""
from __future__ import annotations
from typing import Dict, List, Any

CAREER_SKILLS: Dict[str, List[str]] = {
    "Data Scientist": ["Python", "Machine Learning", "Pandas", "Numpy", "Scikit-Learn", "Statistics", "Sql", "Deep Learning", "Data Analysis"],
    "Data Analyst": ["Sql", "Power Bi", "Excel", "Tableau", "Python", "Data Analysis", "Statistics", "Pandas"],
    "ML Engineer": ["Python", "Deep Learning", "Tensorflow", "Pytorch", "Machine Learning", "Scikit-Learn", "Mlops", "Docker", "Aws"],
    "AI Engineer": ["Python", "Gen Ai", "Llm", "Langchain", "Deep Learning", "Transformers", "Huggingface", "Machine Learning"],
    "Backend Developer": ["Python", "Java", "Sql", "Flask", "Django", "Rest Api", "Docker"],
    "Frontend Developer": ["Html", "Css", "Javascript", "React", "Typescript"],
    "Full Stack Developer": ["Html", "Css", "Javascript", "React", "Node.Js", "Python", "Sql"],
    "DevOps Engineer": ["Aws", "Docker", "Kubernetes", "Ci/Cd", "Linux", "Devops", "Terraform"],
    "Data Engineer": ["Sql", "Python", "Spark", "Hadoop", "Etl", "Data Engineering", "Airflow"],
    "NLP Engineer": ["Python", "Nlp", "Transformers", "Spacy", "Bert", "Huggingface"],
    "Cybersecurity Analyst": ["Cybersecurity", "Linux", "Networking", "Python"],
    "UI/UX Designer": ["Figma", "Ui/Ux", "Html", "Css"],
    "Cloud Engineer": ["Aws", "Azure", "Gcp", "Docker", "Kubernetes", "Linux"],
    "Software Engineer": ["Python", "Java", "Javascript", "Sql", "Git", "Data Structures"],
}

SKILL_ACTIONS: Dict[str, str] = {
    "python": "Complete a Python course and solve coding problems weekly.",
    "sql": "Practice joins, aggregations, and window functions on sample data.",
    "machine learning": "Learn supervised learning and train small scikit-learn models.",
    "deep learning": "Study neural networks with TensorFlow or PyTorch tutorials.",
    "pandas": "Practice data cleaning with Pandas on public CSV datasets.",
    "numpy": "Review NumPy arrays, broadcasting, and numerical operations.",
    "scikit-learn": "Build classification/regression pipelines with scikit-learn.",
    "tensorflow": "Follow TensorFlow beginner tutorials for model training.",
    "pytorch": "Complete a PyTorch intro covering tensors and training loops.",
    "power bi": "Build 2–3 Power BI dashboards using public data.",
    "tableau": "Create interactive dashboards on Tableau Public.",
    "excel": "Learn pivot tables, XLOOKUP, and basic analysis in Excel.",
    "html": "Build static pages with semantic HTML.",
    "css": "Practice responsive layouts with Flexbox and Grid.",
    "javascript": "Learn DOM basics and build small interactive pages.",
    "react": "Build a multi-component React app with hooks.",
    "node.js": "Create a simple REST API with Node.js and Express.",
    "docker": "Containerize a sample app with Docker Compose.",
    "kubernetes": "Complete a beginner lab: pods, deployments, services.",
    "aws": "Study AWS Cloud Practitioner fundamentals and try core services.",
    "azure": "Explore Azure fundamentals and deploy a simple web app.",
    "gcp": "Try Google Cloud free-tier basics (Compute, Storage).",
    "flask": "Build a small Flask API with 2–3 endpoints.",
    "django": "Build a basic Django CRUD application.",
    "rest api": "Design a sample REST API with clear request/response examples.",
    "gen ai": "Practice prompt engineering and LLM API usage.",
    "llm": "Learn how LLMs work at a high level and try API calls.",
    "langchain": "Build a simple RAG or agent demo with LangChain.",
    "transformers": "Use Hugging Face for a text classification demo.",
    "huggingface": "Run a pretrained model from the Hugging Face hub.",
    "nlp": "Practice tokenization and basic NLP tasks.",
    "spacy": "Use spaCy for named entity recognition on sample text.",
    "spark": "Run basic Spark DataFrame transformations.",
    "etl": "Design a simple extract-transform-load pipeline for CSV data.",
    "ci/cd": "Set up a basic GitHub Actions workflow for tests.",
    "linux": "Practice common shell commands and file permissions.",
    "git": "Practice branching, commits, pull requests, and merges.",
    "statistics": "Revise probability, distributions, and hypothesis testing.",
    "data analysis": "Complete an end-to-end analysis project and document findings.",
    "figma": "Design a simple web/mobile UI prototype in Figma.",
}

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _normalize_skill_list(skills: List[str]) -> List[str]:
    out, seen = [], set()
    for s in skills or []:
        n = _norm(str(s))
        if n and n not in seen:
            seen.add(n)
            out.append(str(s).strip())
    return out

def analyze_skill_gap(user_skills: List[str], target_career: str) -> Dict[str, Any]:
    required = CAREER_SKILLS.get(target_career, [])
    if not required:
        tc = _norm(target_career)
        for k in CAREER_SKILLS:
            if tc in _norm(k) or _norm(k) in tc:
                required = CAREER_SKILLS[k]
                break
    user_list = _normalize_skill_list(user_skills)
    user_norm = {_norm(s) for s in user_list}
    matched, missing = [], []
    for s in required:
        rn = _norm(s)
        if rn in user_norm or any(rn in u or u in rn for u in user_norm):
            matched.append(s)
        else:
            missing.append(s)
    total = len(required) if required else 1
    match_pct = round(100.0 * len(matched) / total, 1)
    suggestions = []
    for skill in missing[:8]:
        action = SKILL_ACTIONS.get(_norm(skill), f"Learn the fundamentals of {skill} and complete a small project.")
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
        "summary": f"You match {match_pct}% of skills for {target_career}. {len(matched)} matched, {len(missing)} missing.",
    }

def build_gap_report(user_skills: List[str], top_careers: List[Dict[str, Any]]) -> Dict[str, Any]:
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
