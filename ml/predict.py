"""
Career Prediction + Ranking + Skill Gap Recommendations
"""

import os
import logging
import joblib
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "career_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

_model = None
_vectorizer = None

# Recommended skills for each career (for skill-gap analysis)
CAREER_SKILLS = {
    "Data Scientist": ["Python", "Machine Learning", "Pandas", "Numpy", "Scikit-Learn", "Statistics", "Sql"],
    "Data Analyst": ["Sql", "Power Bi", "Excel", "Tableau", "Python", "Data Analysis", "Statistics"],
    "ML Engineer": ["Python", "Deep Learning", "Tensorflow", "Pytorch", "Machine Learning", "Scikit-Learn"],
    "AI Engineer": ["Python", "Gen Ai", "Llm", "Langchain", "Deep Learning", "Transformers", "Huggingface"],
    "Backend Developer": ["Python", "Java", "Sql", "Flask", "Django", "Rest Api", "Spring Boot"],
    "Frontend Developer": ["Html", "Css", "Javascript", "React", "Typescript", "Vue.Js"],
    "Full Stack Developer": ["Html", "Css", "Javascript", "React", "Node.Js", "Python", "Sql"],
    "DevOps Engineer": ["Aws", "Docker", "Kubernetes", "Ci/Cd", "Linux", "Devops"],
    "Data Engineer": ["Sql", "Python", "Spark", "Hadoop", "Etl", "Data Engineering"],
    "NLP Engineer": ["Python", "Nlp", "Transformers", "Spacy", "Bert", "Huggingface"],
    "Cybersecurity Analyst": ["Cybersecurity", "Linux", "Networking", "Python"],
    "UI/UX Designer": ["Figma", "Ui/Ux"],
    "Project Manager": ["Project Management", "Agile", "Scrum", "Leadership", "Communication"],
    "Digital Marketer": ["Digital Marketing", "Seo", "Content Writing"],
    "Software Engineer": ["Python", "Java", "C++", "Sql", "Git"],
    "Cloud Engineer": ["Aws", "Azure", "Gcp", "Cloud Computing", "Docker"],
    "Android Developer": ["Java", "Kotlin", "Android"],
    "iOS Developer": ["Swift", "Ios"],
}


def load_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError("Model not found. Run ml/train.py first.")
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def prepare_features(skills: str, degree: str, experience: str) -> str:
    skills = skills or ""
    degree = degree or ""
    experience = experience or "0"
    return f"{skills} {degree} {experience} years experience"


def get_skill_gaps(user_skills: List[str], career: str) -> Dict:
    """Return missing skills and match percentage for a career."""
    required = CAREER_SKILLS.get(career, [])
    if not required:
        return {"missing": [], "match_percent": 0, "matched": []}

    user_lower = [s.lower() for s in user_skills]
    matched = [s for s in required if s.lower() in user_lower]
    missing = [s for s in required if s.lower() not in user_lower]
    match_percent = round(len(matched) / len(required) * 100) if required else 0

    return {
        "matched": matched,
        "missing": missing,
        "match_percent": match_percent
    }


def predict_career(skills: str, degree: str = "", experience: str = "0") -> Dict:
    model, vectorizer = load_artifacts()

    feature_text = prepare_features(skills, degree, experience)
    X = vectorizer.transform([feature_text])

    probabilities = model.predict_proba(X)[0]
    classes = model.classes_

    career_probs = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)

    user_skill_list = [s.strip() for s in skills.split(",") if s.strip()]

    top_careers = []
    for career, prob in career_probs[:5]:
        gap = get_skill_gaps(user_skill_list, career)
        top_careers.append({
            "career": career,
            "probability": round(prob * 100, 1),
            "skill_match": gap["match_percent"],
            "missing_skills": gap["missing"][:6],
            "matched_skills": gap["matched"]
        })

    top = top_careers[0]

    return {
        "predicted_career": top["career"],
        "confidence": top["probability"],
        "top_careers": top_careers,
        "user_skills": user_skill_list
    }
