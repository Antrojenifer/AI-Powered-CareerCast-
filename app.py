"""
CareerCast Pro - Professional AI Career Prediction System
Milestone 1: Parsing + Baseline Prediction + Dashboard
"""

import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

from parser.parser import parse_resume, parse_skills_text
from ml.predict import predict_career

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET") or os.environ.get("SECRET_KEY") or "careercast-pro-secret-change-in-production"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"pdf", "docx"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_METRICS = {
    "accuracy": 72.2,
    "precision": 68.5,
    "recall": 72.2,
    "f1_score": 68.5,
    "coverage": 72.2
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        name = request.form.get("name", "").strip() or "Candidate"
        email = request.form.get("email", "").strip()
        degree = request.form.get("degree", "").strip()
        experience = request.form.get("experience", "0").strip() or "0"
        skills_input = request.form.get("skills", "").strip()
        input_mode = request.form.get("input_mode", "skills")

        extracted_skills = []

        # Resume upload mode
        if input_mode == "resume" and "resume" in request.files:
            file = request.files["resume"]
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                result = parse_resume(filepath)
                extracted_skills = result.get("skills", [])
                if not degree:
                    degree = result.get("education", "")
                if not experience or experience == "0":
                    experience = result.get("experience", "0") or "0"
                if name == "Candidate" and result.get("name"):
                    name = result.get("name")

                try:
                    os.remove(filepath)
                except OSError:
                    pass

        # Paste skills mode
        if skills_input:
            pasted = parse_skills_text(skills_input)
            # Merge unique
            all_skills = list(dict.fromkeys(extracted_skills + pasted))
            extracted_skills = all_skills

        if not extracted_skills and not skills_input:
            flash("Please upload a resume or paste your skills.", "warning")
            return redirect(url_for("index"))

        skills_str = ", ".join(extracted_skills) if extracted_skills else skills_input

        # Run prediction
        prediction = predict_career(
            skills=skills_str,
            degree=degree,
            experience=experience
        )

        context = {
            "name": name,
            "email": email,
            "degree": degree,
            "experience": experience,
            "skills": prediction["user_skills"] or extracted_skills,
            "predicted_career": prediction["predicted_career"],
            "confidence": prediction["confidence"],
            "top_careers": prediction["top_careers"],
            "metrics": MODEL_METRICS
        }

        return render_template("dashboard.html", **context)

    except FileNotFoundError:
        flash("Model not found. Please train the model first.", "danger")
        return redirect(url_for("index"))
    except Exception as e:
        logger.error(f"Error: {e}")
        flash("Something went wrong. Please try again.", "danger")
        return redirect(url_for("index"))


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "app": "CareerCast Pro"}), 200


@app.errorhandler(413)
def too_large(e):
    flash("File too large. Max 5 MB allowed.", "danger")
    return redirect(url_for("index")), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
