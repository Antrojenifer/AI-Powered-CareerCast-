"""
AI-Powered Career Intelligence Platform
Milestone 1: Parsing + Baseline Prediction
Milestone 2: Advanced ML (RF, XGBoost), Top-K Ranking, Skill Alignment
"""

import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

from parser.parser import parse_resume, parse_skills_text
from ml.predict import predict_career, load_metrics, load_model_comparison

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
        top_k = int(request.form.get("top_k", 5) or 5)
        model_choice = request.form.get("model_choice", "").strip() or None

        extracted_skills = []

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

        if skills_input:
            pasted = parse_skills_text(skills_input)
            extracted_skills = list(dict.fromkeys(extracted_skills + pasted))

        if not extracted_skills and not skills_input:
            flash("Please upload a resume or paste your skills.", "warning")
            return redirect(url_for("index"))

        skills_str = ", ".join(extracted_skills) if extracted_skills else skills_input

        prediction = predict_career(
            skills=skills_str,
            degree=degree,
            experience=experience,
            top_k=top_k,
            model_name=model_choice,
        )

        metrics = load_metrics()
        comparison = load_model_comparison()

        context = {
            "name": name,
            "email": email,
            "degree": degree,
            "experience": experience,
            "skills": prediction["user_skills"] or extracted_skills,
            "predicted_career": prediction["predicted_career"],
            "confidence": prediction["confidence"],
            "top_careers": prediction["top_careers"],
            "metrics": metrics,
            "model_used": prediction.get("model_used", "logistic_regression"),
            "feature_type": prediction.get("feature_type", "tfidf"),
            "skill_alignment": prediction.get("skill_alignment", 0),
            "model_comparison": comparison,
            "milestone": 2,
        }

        return render_template("dashboard.html", **context)

    except FileNotFoundError:
        flash("Model not found. Please train the model first (python -m ml.train_advanced).", "danger")
        return redirect(url_for("index"))
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        flash("Something went wrong. Please try again.", "danger")
        return redirect(url_for("index"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API for Milestone 2 programmatic access."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        skills = data.get("skills", "")
        degree = data.get("degree", "")
        experience = str(data.get("experience", "0"))
        top_k = int(data.get("top_k", 5))
        model_name = data.get("model_name")

        if not skills:
            return jsonify({"error": "skills required"}), 400

        result = predict_career(
            skills=skills,
            degree=degree,
            experience=experience,
            top_k=top_k,
            model_name=model_name,
        )
        result["metrics"] = load_metrics()
        result["model_comparison"] = load_model_comparison()
        return jsonify(result)
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    metrics = load_metrics()
    return jsonify({
        "status": "healthy",
        "app": "AI-Powered Career Intelligence Platform",
        "milestone": 2,
        "best_model": metrics.get("model_name", "unknown"),
    }), 200


@app.errorhandler(413)
def too_large(e):
    flash("File too large. Max 5 MB allowed.", "danger")
    return redirect(url_for("index")), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
