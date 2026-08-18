from ml.skill_gap import analyze_skill_gap, build_gap_report

def test_analyze_skill_gap_basic():
    r = analyze_skill_gap(["Python", "SQL", "Pandas"], "Data Analyst")
    assert "match_percent" in r
    assert isinstance(r["matched_skills"], list)

def test_build_gap_report():
    report = build_gap_report(["Python", "Machine Learning"], [{"career": "Data Scientist", "confidence": 80.0}])
    assert report["primary_career"] == "Data Scientist"
