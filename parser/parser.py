"""
Resume Parser Module - CareerCast Pro
Extracts skills, education, role, experience using SpaCy + rule matching.
"""

import os
import re
import logging
from typing import Dict, List

import pdfplumber
from docx import Document
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    logger.warning("SpaCy model not loaded. Run: python -m spacy download en_core_web_sm")

SKILLS_PATH = os.path.join(os.path.dirname(__file__), "skills.csv")

def load_skills() -> List[str]:
    try:
        df = pd.read_csv(SKILLS_PATH)
        return list(set(df["skill"].str.lower().str.strip().tolist()))
    except Exception as e:
        logger.error(f"Error loading skills: {e}")
        return []

KNOWN_SKILLS = load_skills()


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"PDF error: {e}")
    return text


def extract_text_from_docx(file_path: str) -> str:
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logger.error(f"DOCX error: {e}")
    return text


def extract_skills(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for skill in KNOWN_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill.title())
    return list(dict.fromkeys(found))


def extract_education(text: str) -> str:
    patterns = [
        r"(B\.?Tech|Bachelor of Technology|B\.E\.?)[^\n,]{0,50}",
        r"(M\.?Tech|Master of Technology|M\.E\.?)[^\n,]{0,50}",
        r"(B\.?Sc|Bachelor of Science)[^\n,]{0,50}",
        r"(M\.?Sc|Master of Science)[^\n,]{0,50}",
        r"(MBA|Master of Business Administration)[^\n,]{0,50}",
        r"(B\.?Des|Bachelor of Design)[^\n,]{0,50}",
        r"(BBA|Bachelor of Business Administration)[^\n,]{0,50}",
        r"(Ph\.?D|Doctor of Philosophy)[^\n,]{0,50}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return ""


def extract_role(text: str) -> str:
    patterns = [
        r"(?:Senior|Junior|Lead)?\s*(?:Software|Data|ML|Machine Learning|Full Stack|Frontend|Backend|DevOps|Cloud|AI|NLP|Android|iOS|UI/UX|Cybersecurity|Project)\s+(?:Engineer|Developer|Scientist|Analyst|Manager|Designer)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return ""


def extract_experience(text: str) -> str:
    patterns = [
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience",
        r"experience\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)",
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:in|as)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def extract_name(text: str) -> str:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    if len(first.split()) <= 4 and not re.search(r"\d|@|http|linkedin|github|resume|cv", first, re.IGNORECASE):
        return first
    return ""


def parse_resume(file_path: str) -> Dict:
    result = {
        "name": "",
        "skills": [],
        "education": "",
        "role": "",
        "experience": ""
    }

    if not os.path.exists(file_path):
        return result

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        text = extract_text_from_docx(file_path)
    else:
        return result

    if not text.strip():
        return result

    result["name"] = extract_name(text)
    result["skills"] = extract_skills(text)
    result["education"] = extract_education(text)
    result["role"] = extract_role(text)
    result["experience"] = extract_experience(text)
    return result


def parse_skills_text(skills_text: str) -> List[str]:
    """Parse free-text skills input (comma or space separated)."""
    if not skills_text:
        return []
    # Split by comma or newline
    raw = re.split(r"[,\n]+", skills_text.lower())
    cleaned = []
    for item in raw:
        item = item.strip()
        if not item:
            continue
        # Match against known skills
        for skill in KNOWN_SKILLS:
            if skill in item or item in skill:
                cleaned.append(skill.title())
                break
        else:
            # Keep original if not in list
            cleaned.append(item.title())
    return list(dict.fromkeys(cleaned))
