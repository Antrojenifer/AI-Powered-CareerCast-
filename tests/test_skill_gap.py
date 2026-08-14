from ml.skill_gap import analyze_skill_gap, build_gap_report


def test_analyze_skill_gap_basic():
    result = analyze_skill_gap(
        ["Python", "SQL", "Pandas"],
        "Data Analyst",
    )
    assert "match_percent" in result
    assert isinstance(result["matched_skills"], list)
    assert isinstance(result["missing_skills"], list)
    assert result["match_percent"] >= 0


def test_build_gap_report():
    report = build_gap_report(
        ["Python", "Machine Learning"],
        [{"career": "Data Scientist", "confidence": 80.0}],
    )
    assert report["primary_career"] == "Data Scientist"
    assert "career_gaps" in report
