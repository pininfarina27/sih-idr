import requests
import json
import os

def fetch_osm():
    # Bounding boxes covering S1, S2, S3a in Coventry, UK
    # S1: 52.401 to 52.408, -1.526 to -1.505
    # S2: 52.402 to 52.404, -1.570 to -1.556
    # S3a: 52.406 to 52.408, -1.495 to -1.471
    bbox = "52.395,-1.575,52.415,-1.465"
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"]({bbox});
    );
    out geom;
    """
    mirrors = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass-api.de/api/interpreter"
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*"
    }
    
    data = None
    for m in mirrors:
        print(f"Trying mirror {m}...")
        try:
            resp = requests.post(m, data={"data": query}, headers=headers, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                print(f"Success from {m}!")
                break
            else:
                print(f"Mirror returned status {resp.status_code}")
        except Exception as e:
            print(f"Mirror failed: {e}")

    if not data or "elements" not in data:
        print("Failed to fetch from Overpass mirrors.")
        return False

    elements = data["elements"]
    features = []
    for el in elements:
        if "geometry" in el and len(el["geometry"]) > 1:
            coords = [[p["lon"], p["lat"]] for p in el["geometry"]]
            features.append({
                "type": "Feature",
                "properties": {
                    "id": el["id"],
                    "highway": el.get("tags", {}).get("highway", "road"),
                    "name": el.get("tags", {}).get("name", "unnamed")
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    out_path = "../public/data/osm_roads.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)

    file_size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved {len(features)} independent OSM road lines to {out_path} ({file_size_kb:.1f} KB)")
    return True

if __name__ == "__main__":
    fetch_osm()
