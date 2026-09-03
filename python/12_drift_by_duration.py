import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("../results", exist_ok=True)

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    R = 6378137.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((phi2-phi1)/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(np.radians((lon2-lon1)/2))**2
    return 2*R*np.arcsin(np.sqrt(a))

def path_length(pts):
    total = 0.0
    for i in range(1, len(pts)):
        total += haversine(pts[i-1]["lat"], pts[i-1]["lon"], pts[i]["lat"], pts[i]["lon"])
    return total

with open("../public/data/segments.json") as f:
    meta = {s["id"]: s for s in json.load(f)["segments"]}

segments = ["S1", "S2", "S3a"]
# We test how many seconds INTO the blackout we measure drift
offsets = [10, 20, 30, 40]

report_lines = [
    "=" * 70,
    "DRIFT vs BLACKOUT DURATION ANALYSIS",
    "Measured at GPS reacquisition after N seconds of complete GPS blackout",
    "ISRO Target: AI drift < 10% of distance traveled during blackout",
    "=" * 70,
    f"{'Segment':<8} {'Into BKT':>9} {'Road Dist':>11} {'Raw Drift':>11} {'AI Drift':>10} {'Raw%':>7} {'AI%':>6} {'Pass?':>7}",
    "-" * 70,
]

all_results = {}
for seg in segments:
    gt  = json.load(open(f"../public/data/segment_{seg}_gt.json"))
    raw = json.load(open(f"../public/data/segment_{seg}_raw_dr.json"))
    ai  = json.load(open(f"../public/data/segment_{seg}_ai_fused.json"))
    
    bo_start = meta[seg]["blackout_start_ts"]
    bo_end   = meta[seg]["blackout_end_ts"]
    
    # GT point at blackout start (last good GPS before blackout)
    gt_at_start = min(gt, key=lambda p: abs(p["ts"] - bo_start))
    
    all_results[seg] = {}
    for dur in offsets:
        # Only test if we have enough blackout
        if bo_start + dur > bo_end + 5:
            continue
        
        t_query = bo_start + dur
        
        # GT reference position AT that moment
        gt_at_query = min(gt, key=lambda p: abs(p["ts"] - t_query))
        
        # DR position at same time
        raw_at_query = min(raw, key=lambda p: abs(p["ts"] - t_query))
        ai_at_query  = min(ai,  key=lambda p: abs(p["ts"] - t_query))
        
        # Distance the car actually traveled during the blackout window
        gt_window = [p for p in gt if bo_start <= p["ts"] <= t_query]
        road_dist = path_length(gt_window) if len(gt_window) > 1 else 1.0
        
        raw_drift = haversine(gt_at_query["lat"], gt_at_query["lon"],
                              raw_at_query["lat"], raw_at_query["lon"])
        ai_drift  = haversine(gt_at_query["lat"], gt_at_query["lon"],
                              ai_at_query["lat"],  ai_at_query["lon"])
        
        raw_pct = (raw_drift / road_dist * 100) if road_dist > 0 else 0
        ai_pct  = (ai_drift  / road_dist * 100) if road_dist > 0 else 0
        passed  = "YES" if ai_pct < 10 else "NO"
        
        all_results[seg][str(dur)] = {
            "road_dist_m": round(road_dist, 1),
            "raw_drift_m": round(raw_drift, 1), "ai_drift_m": round(ai_drift, 1),
            "raw_pct": round(raw_pct, 2), "ai_pct": round(ai_pct, 2),
            "isro_pass": passed == "YES"
        }
        report_lines.append(
            f"{seg:<8} {dur:>7}s   {road_dist:>9.1f}m  {raw_drift:>9.1f}m  {ai_drift:>8.1f}m  {raw_pct:>5.1f}%  {ai_pct:>4.1f}%  {passed:>7}"
        )
    report_lines.append("")

report = "\n".join(report_lines)
print(report)
with open("../results/drift_by_duration.txt", "w") as f:
    f.write(report)

with open("../public/data/drift_results.json", "w") as f:
    json.dump({"segments": all_results, "durations": offsets}, f, indent=2)

# Chart
colors = {"S1": "#6366F1", "S2": "#10B981", "S3a": "#F59E0B"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for seg in segments:
    res = all_results.get(seg, {})
    durs = sorted(int(k) for k in res.keys())
    if not durs: continue
    raw_pcts = [res[str(d)]["raw_pct"] for d in durs]
    ai_pcts  = [res[str(d)]["ai_pct"]  for d in durs]
    axes[0].plot(durs, raw_pcts, "o--", color=colors[seg], label=seg, linewidth=2, markersize=7)
    axes[1].plot(durs, ai_pcts,  "o-",  color=colors[seg], label=seg, linewidth=2, markersize=7)

for ax, title in zip(axes, ["Raw DR Drift %", "AI-ML Fused Drift %"]):
    ax.axhline(10, color="red", linestyle="--", linewidth=2, label="ISRO 10% target")
    ax.set_xlabel("Seconds into GPS Blackout", fontsize=11)
    ax.set_ylabel("Position Drift %", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle("Drift% vs GPS Blackout Duration — IO-VNBD Benchmark", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../results/drift_vs_duration.png", dpi=150)
plt.close()
print("Saved drift_vs_duration.png")
