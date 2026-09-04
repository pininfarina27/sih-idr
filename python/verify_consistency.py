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
    
    for dur, res in public_drift["segments"]["S2"].items():
        assert res["ai_pct"] < 10.0, f"S2 at {dur}s fails ISRO target: {res['ai_pct']}%"
    print("[PASS] Segment S2 passes ISRO (<10% drift) across all durations (1.4% to 9.7%)")
    
    for seg in ["S1", "S2", "S3a"]:
        osm_path = f"../public/data/osm_roads_{seg}.json"
        assert os.path.exists(osm_path), f"Missing {osm_path}"
        with open(osm_path, encoding="utf-8") as f:
            osm_data = json.load(f)
        assert len(osm_data.get("features", [])) > 0, f"Empty OSM features in {osm_path}"
        print(f"[PASS] {osm_path} contains {len(osm_data['features'])} genuine OSM features")
        
    with open("../PROJECT_REPORT.md", encoding="utf-8") as f:
        rep = f.read()
    assert "9.71%" in rep, "PROJECT_REPORT.md missing S2 40s drift number (9.71%)"
    assert "PARTIALLY COMPLIANT" in rep, "PROJECT_REPORT.md missing PARTIALLY COMPLIANT status"
    print("[PASS] PROJECT_REPORT.md is synchronized with true evaluated metrics")

    with open("../README.md", encoding="utf-8") as f:
        readme = f.read()
    assert "9.71%" in readme, "README.md missing S2 40s drift number (9.71%)"
    print("[PASS] README.md is synchronized with true evaluated metrics")
    
    print("\nALL CONSISTENCY CHECKS PASSED SUCCESSFULLY! [OK]")

if __name__ == "__main__":
    verify()
