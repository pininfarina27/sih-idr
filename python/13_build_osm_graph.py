"""
13_build_osm_graph.py — Phase 2: OSM Road Graph Builder
========================================================
Converts existing osm_roads_{sid}.json GeoJSON files into directed road graphs
suitable for the Road-Constrained Particle Filter (RCPF) in Phase 3.

Output: public/data/road_graph_{sid}.json
Schema:
  {
    "nodes": { "<node_id>": {"lat": float, "lon": float} },
    "edges": [
      {
        "id":         int,
        "start_node": str,
        "end_node":   str,
        "start_lat":  float, "start_lon": float,
        "end_lat":    float, "end_lon":   float,
        "azimuth_deg":float,   # 0=N, 90=E, 180=S, 270=W (forward direction)
        "rev_azimuth":float,   # reverse direction azimuth
        "length_m":   float,
        "highway":    str,
        "coords":     [[lon,lat], ...],  # full polyline for particle interpolation
        "drivable":   bool
      }
    ]
  }
"""

import json
import math
import os

# --------------------------------------------------------------------------
# Drivable highway types (vehicles can use these)
# --------------------------------------------------------------------------
DRIVABLE = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified",
    "residential",
    "service",
    "living_street",
    "road",
}

# --------------------------------------------------------------------------
# Haversine distance in metres between two (lat,lon) pairs
# --------------------------------------------------------------------------
_R = 6_378_137.0

def haversine(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * _R * math.asin(math.sqrt(a))

# --------------------------------------------------------------------------
# Bearing from point A to point B  (0=North, clockwise degrees)
# --------------------------------------------------------------------------
def bearing(lat1, lon1, lat2, lon2):
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

# --------------------------------------------------------------------------
# Snap nearby endpoints to the same node (within SNAP_DIST metres)
# --------------------------------------------------------------------------
SNAP_DIST = 5.0   # metres

def make_node_id(lat, lon):
    """Round to ~0.11m resolution for node deduplication."""
    return f"{round(lat, 6)},{round(lon, 6)}"

def snap_to_existing(lat, lon, nodes_latlon, snap_dist=SNAP_DIST):
    """Return existing node_id if within snap_dist, else None."""
    for nid, (nlat, nlon) in nodes_latlon.items():
        if haversine(lat, lon, nlat, nlon) <= snap_dist:
            return nid
    return None

# --------------------------------------------------------------------------
# Build graph for one segment
# --------------------------------------------------------------------------
def build_graph(sid):
    osm_path = f"../public/data/osm_roads_{sid}.json"
    out_path  = f"../public/data/road_graph_{sid}.json"

    print(f"\n=== Building road graph for {sid} ===")
    with open(osm_path) as f:
        geojson = json.load(f)

    features = geojson["features"]
    print(f"  Input features: {len(features)}")

    # Filter to drivable roads only
    drivable_feats = [feat for feat in features
                      if feat.get("properties", {}).get("highway", "") in DRIVABLE]
    print(f"  Drivable features: {len(drivable_feats)}")

    # Build node registry and edges
    nodes_latlon = {}   # node_id -> (lat, lon)
    edges        = []
    edge_id      = 0

    for feat in drivable_feats:
        coords  = feat["geometry"]["coordinates"]   # [[lon, lat], ...]
        highway = feat.get("properties", {}).get("highway", "unknown")
        name    = feat.get("properties", {}).get("name", "")

        if len(coords) < 2:
            continue

        # --- Register start and end nodes with snapping ---
        s_lon, s_lat = coords[0]
        e_lon, e_lat = coords[-1]

        # Start node
        s_nid = snap_to_existing(s_lat, s_lon, nodes_latlon)
        if s_nid is None:
            s_nid = make_node_id(s_lat, s_lon)
            nodes_latlon[s_nid] = (s_lat, s_lon)

        # End node
        e_nid = snap_to_existing(e_lat, e_lon, nodes_latlon)
        if e_nid is None:
            e_nid = make_node_id(e_lat, e_lon)
            nodes_latlon[e_nid] = (e_lat, e_lon)

        # Compute edge length (sum of all segment lengths)
        length_m = 0.0
        for i in range(len(coords) - 1):
            la1, lo1 = coords[i][1],   coords[i][0]
            la2, lo2 = coords[i+1][1], coords[i+1][0]
            length_m += haversine(la1, lo1, la2, lo2)

        if length_m < 0.1:   # skip degenerate edges
            continue

        # Azimuth: bearing from start to end (overall road direction)
        fwd_az  = bearing(s_lat, s_lon, e_lat, e_lon)
        rev_az  = (fwd_az + 180) % 360

        # Store coords as [lat, lon] pairs for easy particle interpolation
        latlon_coords = [[c[1], c[0]] for c in coords]

        edge_base = {
            "id":          edge_id,
            "start_node":  s_nid,
            "end_node":    e_nid,
            "start_lat":   s_lat,
            "start_lon":   s_lon,
            "end_lat":     e_lat,
            "end_lon":     e_lon,
            "azimuth_deg": round(fwd_az, 2),
            "rev_azimuth": round(rev_az, 2),
            "length_m":    round(length_m, 3),
            "highway":     highway,
            "name":        name,
            "coords":      latlon_coords,
        }
        edges.append(edge_base)
        edge_id += 1

        # Add reverse edge (two-way roads — OSM default unless oneway=yes)
        # We always add both directions; RCPF heading constraint will filter
        rev_edge = {
            "id":          edge_id,
            "start_node":  e_nid,
            "end_node":    s_nid,
            "start_lat":   e_lat,
            "start_lon":   e_lon,
            "end_lat":     s_lat,
            "end_lon":     s_lon,
            "azimuth_deg": round(rev_az, 2),
            "rev_azimuth": round(fwd_az, 2),
            "length_m":    round(length_m, 3),
            "highway":     highway,
            "name":        name,
            "coords":      list(reversed(latlon_coords)),
        }
        edges.append(rev_edge)
        edge_id += 1

    # Build adjacency list for connectivity check
    adjacency = {}
    for edge in edges:
        sn = edge["start_node"]
        en = edge["end_node"]
        adjacency.setdefault(sn, []).append(edge["id"])
        adjacency.setdefault(en, [])   # ensure end node exists too

    # Nodes dict
    nodes_dict = {nid: {"lat": lat, "lon": lon} for nid, (lat, lon) in nodes_latlon.items()}

    graph = {
        "segment":    sid,
        "nodes":      nodes_dict,
        "edges":      edges,
        "adjacency":  {k: v for k, v in adjacency.items()},
    }

    print(f"  Nodes: {len(nodes_dict)}")
    print(f"  Edges (bidirectional): {len(edges)}")

    with open(out_path, "w") as f:
        json.dump(graph, f, separators=(",", ":"))   # compact for file size

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved {out_path}  ({size_kb:.1f} KB)")
    return graph


# --------------------------------------------------------------------------
# Validate: snap blackout entry GPS to nearest edge; check < 5m
# --------------------------------------------------------------------------
def point_to_segment_dist(plat, plon, lat1, lon1, lat2, lon2):
    """
    Perpendicular distance from point P to line segment (A->B) in metres.
    Returns (dist_m, fraction_along_segment).
    """
    # Project P onto line AB using dot product in local metres
    dx  = (lon2 - lon1) * math.cos(math.radians((lat1+lat2)/2)) * 111319.5
    dy  = (lat2 - lat1) * 111319.5
    seg_len_sq = dx*dx + dy*dy
    if seg_len_sq < 1e-10:
        return haversine(plat, plon, lat1, lon1), 0.0

    px = (plon - lon1) * math.cos(math.radians((lat1+lat2)/2)) * 111319.5
    py = (plat - lat1) * 111319.5

    t = max(0.0, min(1.0, (px*dx + py*dy) / seg_len_sq))
    closest_lat = lat1 + t * (lat2 - lat1)
    closest_lon = lon1 + t * (lon2 - lon1)
    return haversine(plat, plon, closest_lat, closest_lon), t


def snap_point_to_graph(lat, lon, graph, max_dist=50.0):
    """Return (edge_id, dist_m, along_m) of nearest edge."""
    best_dist = float("inf")
    best_edge = None
    best_along = 0.0

    for edge in graph["edges"]:
        coords = edge["coords"]   # [[lat, lon], ...]
        for i in range(len(coords) - 1):
            la1, lo1 = coords[i]
            la2, lo2 = coords[i+1]
            d, t = point_to_segment_dist(lat, lon, la1, lo1, la2, lo2)
            if d < best_dist:
                best_dist = d
                best_edge = edge["id"]
                best_along = t * haversine(la1, lo1, la2, lo2)

    return best_edge, best_dist, best_along


def validate_graph(sid, graph):
    import pandas as pd
    df = pd.read_csv(f"data/{sid}_features.csv")
    bk = df[df["blackout"] == True]

    entry_lat = bk.iloc[0]["lat"]
    entry_lon = bk.iloc[0]["lon"]
    exit_lat  = bk.iloc[-1]["lat"]
    exit_lon  = bk.iloc[-1]["lon"]

    entry_edge, entry_dist, entry_along = snap_point_to_graph(entry_lat, entry_lon, graph)
    exit_edge,  exit_dist,  exit_along  = snap_point_to_graph(exit_lat,  exit_lon,  graph)

    print(f"\n  Validation {sid}:")
    print(f"    Entry ({entry_lat:.5f}, {entry_lon:.5f}) -> edge {entry_edge}, dist={entry_dist:.1f}m")
    print(f"    Exit  ({exit_lat:.5f}, {exit_lon:.5f}) -> edge {exit_edge},  dist={exit_dist:.1f}m")

    if entry_dist <= 15.0:
        print(f"    [PASS] Entry snaps within {entry_dist:.1f}m (<= 15m)")
    else:
        print(f"    [WARN] Entry snap dist {entry_dist:.1f}m > 15m  (check OSM coverage)")

    if exit_dist <= 15.0:
        print(f"    [PASS] Exit snaps within {exit_dist:.1f}m (<= 15m)")
    else:
        print(f"    [WARN] Exit snap dist {exit_dist:.1f}m > 15m")

    return entry_dist, exit_dist


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    results = {}
    for sid in ["S1", "S2", "S3a"]:
        graph = build_graph(sid)
        e_dist, x_dist = validate_graph(sid, graph)
        results[sid] = {"entry_dist_m": round(e_dist, 2), "exit_dist_m": round(x_dist, 2),
                        "nodes": len(graph["nodes"]), "edges": len(graph["edges"])}

    print("\n=== SUMMARY ===")
    for sid, r in results.items():
        print(f"  {sid}: {r['nodes']} nodes, {r['edges']} edges, "
              f"entry={r['entry_dist_m']}m exit={r['exit_dist_m']}m")
    print("\nPhase 2 complete — road graphs ready for RCPF (Phase 3)")
