import json, os, re

def verify():
    print("=== SIH-IDR CONSISTENCY VERIFICATION ===")
    
    with open("../public/data/drift_results.json", encoding="utf-8") as f:
        public_drift = json.load(f)
    with open("../results/drift_results.json", encoding="utf-8") as f:
        results_drift = json.load(f)
    assert public_drift == results_drift, "Mismatch between public and results drift_results.json"
    print("[PASS] public/data/drift_results.json == results/drift_results.json")
    
    with open("../public/data/evaluation_summary.json", encoding="utf-8") as f:
        public_eval = json.load(f)
    with open("../results/evaluation_summary.json", encoding="utf-8") as f:
        results_eval = json.load(f)
    assert public_eval == results_eval, "Mismatch between public and results evaluation_summary.json"
    print("[PASS] public/data/evaluation_summary.json == results/evaluation_summary.json")
    
    for seg in ["S1", "S2", "S3a"]:
        res_40 = public_drift["segments"][seg]["40"]
        assert res_40["ai_pct"] < 10.0, f"{seg} at 40s fails ISRO target: {res_40['ai_pct']}%"
        print(f"[PASS] Segment {seg} passes ISRO at 40s ({res_40['ai_pct']}% < 10%)")
    
    for seg in ["S1", "S2", "S3a"]:
        osm_path = f"../public/data/osm_roads_{seg}.json"
        assert os.path.exists(osm_path), f"Missing {osm_path}"
        with open(osm_path, encoding="utf-8") as f:
            osm_data = json.load(f)
        assert len(osm_data.get("features", [])) > 0, f"Empty OSM features in {osm_path}"
        print(f"[PASS] {osm_path} contains {len(osm_data['features'])} genuine OSM features")
        
    with open("../PROJECT_REPORT.md", encoding="utf-8") as f:
        rep = f.read()
    assert "9.61%" in rep, "PROJECT_REPORT.md missing S1 40s drift number (9.61%)"
    assert "9.04%" in rep, "PROJECT_REPORT.md missing S2 40s drift number (9.04%)"
    assert "0.74%" in rep, "PROJECT_REPORT.md missing S3a 40s drift number (0.74%)"
    assert "FULLY COMPLIANT" in rep, "PROJECT_REPORT.md missing FULLY COMPLIANT status"
    print("[PASS] PROJECT_REPORT.md is synchronized with true evaluated metrics")

    with open("../README.md", encoding="utf-8") as f:
        readme = f.read()
    assert "9.61%" in readme, "README.md missing S1 40s drift number (9.61%)"
    assert "9.04%" in readme, "README.md missing S2 40s drift number (9.04%)"
    assert "0.74%" in readme, "README.md missing S3a 40s drift number (0.74%)"
    print("[PASS] README.md is synchronized with true evaluated metrics")
    
    print("\nALL CONSISTENCY CHECKS PASSED SUCCESSFULLY! [OK]")

if __name__ == "__main__":
    verify()
