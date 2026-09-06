"""
14_rcpf.py -- Phase 3: Road-Constrained Particle Filter (v3)
=============================================================
Replaces open-space dead reckoning during GPS blackout.
Road azimuth (OSM) replaces broken gyroscope as heading.
Lateral acceleration disambiguates turns.

v3 changes
----------
- Global reference heading: particles penalized if they diverge
  more than MAX_HEAD_DEV from the cumulative heading estimate.
- Stronger straight-road prior (5.0x when no turn signal).
- predict() now accepts global_heading_deg for reference weighting.
- heading_update(): external heading tracker using RCPF output.
"""

import math
import random
import numpy as np

_R = 6_378_137.0


def haversine(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * _R * math.asin(math.sqrt(max(0.0, a)))


def angle_diff(a, b):
    """Signed CW angle (a - b) in degrees, range (-180, 180]."""
    d = (a - b + 360.0) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def point_on_edge(edge, along_m):
    """Interpolate (lat, lon) at distance along_m from edge start."""
    coords = edge["coords"]
    remaining = max(0.0, along_m)
    for i in range(len(coords) - 1):
        la1, lo1 = coords[i]
        la2, lo2 = coords[i + 1]
        seg_len = haversine(la1, lo1, la2, lo2)
        if remaining <= seg_len or i == len(coords) - 2:
            t = (remaining / seg_len) if seg_len > 1e-6 else 0.0
            t = min(1.0, t)
            return la1 + t * (la2 - la1), lo1 + t * (lo2 - lo1)
        remaining -= seg_len
    return coords[-1][0], coords[-1][1]


# ---------------------------------------------------------------------------
RESAMPLE_EVERY   = 50
N_PARTICLES      = 500
INIT_SNAP_RADIUS = 30.0     # m
INIT_HEAD_TOL    = 45.0     # deg
MAX_HEAD_DEV     = 60.0     # deg -- max global heading deviation penalty
TURN_STRONG      = 1.0      # m/s^2
TURN_SOFT        = 0.5      # m/s^2
UTURN_WEIGHT     = 0.001


class RCParticleFilter:
    """Road-Constrained Particle Filter — particle: [edge_id, along_m, weight]"""

    def __init__(self, graph, N=N_PARTICLES, rng_seed=42):
        self.graph    = graph
        self.N        = N
        self.edges    = graph["edges"]
        self.adj      = graph["adjacency"]
        self.rng      = random.Random(rng_seed)
        self.np_rng   = np.random.default_rng(rng_seed)
        self.edge_map = {e["id"]: e for e in self.edges}

    # -----------------------------------------------------------------------
    def initialize(self, lat, lon, heading_deg):
        """Distribute N particles on edges near (lat,lon) with heading-weighted init."""
        candidates = []
        for edge in self.edges:
            mid_lat = (edge["start_lat"] + edge["end_lat"]) / 2
            mid_lon = (edge["start_lon"] + edge["end_lon"]) / 2
            if abs(mid_lat - lat) > 0.0015 or abs(mid_lon - lon) > 0.003:
                continue
            dist, t = self._point_to_edge_dist(lat, lon, edge)
            if dist > INIT_SNAP_RADIUS:
                continue
            head_diff = abs(angle_diff(heading_deg, edge["azimuth_deg"]))
            if head_diff > INIT_HEAD_TOL:
                continue
            candidates.append((edge["id"], t * edge["length_m"], dist, head_diff))

        if not candidates:
            for edge in self.edges:
                mid_lat = (edge["start_lat"] + edge["end_lat"]) / 2
                mid_lon = (edge["start_lon"] + edge["end_lon"]) / 2
                if abs(mid_lat - lat) > 0.0015 or abs(mid_lon - lon) > 0.003:
                    continue
                dist, t = self._point_to_edge_dist(lat, lon, edge)
                if dist > INIT_SNAP_RADIUS:
                    continue
                head_diff = abs(angle_diff(heading_deg, edge["azimuth_deg"]))
                if head_diff > 60.0:
                    continue
                candidates.append((edge["id"], t * edge["length_m"], dist, head_diff))

        if not candidates:
            best_e = min(self.edges,
                         key=lambda e: haversine(lat, lon,
                                                 (e["start_lat"] + e["end_lat"]) / 2,
                                                 (e["start_lon"] + e["end_lon"]) / 2))
            dist, t = self._point_to_edge_dist(lat, lon, best_e)
            candidates = [(best_e["id"], t * best_e["length_m"], dist, 0.0)]

        # Exponential heading weight: strong bias toward correct heading
        raw_w = [(1.0 / (d + 0.5)) * math.exp(-hd / 15.0) for _, _, d, hd in candidates]
        total  = sum(raw_w)
        norm_w = [w / total for w in raw_w]

        particles = []
        for _ in range(self.N):
            idx = self._weighted_choice_py(norm_w)
            eid, along, _, _ = candidates[idx]
            edge    = self.edge_map[eid]
            jitter  = self.rng.gauss(0.0, 0.5)
            along_j = max(0.0, min(edge["length_m"], along + jitter))
            particles.append([eid, along_j, 1.0 / self.N])

        return particles

    # -----------------------------------------------------------------------
    def predict(self, particles, speed_ms, dt, lat_accel, global_heading=None):
        """
        Advance particles by speed_ms * dt metres.
        global_heading: reference compass heading (degrees) from cumulative
                        weighted position estimate. Used to down-weight particles
                        on roads that deviate > MAX_HEAD_DEV from this reference.
        """
        advance = max(0.0, speed_ms * dt)
        new_particles = []

        for eid, along_m, weight in particles:
            curr_edge  = self.edge_map.get(eid)
            if curr_edge is None:
                new_particles.append([eid, along_m, weight])
                continue

            remaining  = advance
            curr_along = along_m

            for _ in range(8):
                space_ahead = curr_edge["length_m"] - curr_along
                if remaining <= space_ahead:
                    curr_along += remaining
                    break
                remaining -= space_ahead
                next_eid = self._transition(curr_edge, lat_accel, global_heading)
                if next_eid is None:
                    curr_along = curr_edge["length_m"]
                    break
                curr_edge  = self.edge_map[next_eid]
                curr_along = 0.0

            # Apply global heading penalty to particle weight
            new_w = weight
            if global_heading is not None:
                edge_az  = curr_edge["azimuth_deg"]
                head_dev = abs(angle_diff(edge_az, global_heading))
                if head_dev > MAX_HEAD_DEV:
                    # Exponential penalty for large deviations
                    new_w = weight * math.exp(-(head_dev - MAX_HEAD_DEV) / 30.0)

            new_particles.append([curr_edge["id"], curr_along, new_w])

        return new_particles

    def _transition(self, curr_edge, lat_accel, global_heading=None):
        """Sample next edge. Uses turn weight + optional global heading reference."""
        end_node = curr_edge["end_node"]
        curr_az  = curr_edge["azimuth_deg"]

        candidate_eids = self.adj.get(end_node, [])
        forward = []
        for ceid in candidate_eids:
            cedge = self.edge_map.get(ceid)
            if cedge is None or cedge["start_node"] != end_node:
                continue
            delta = angle_diff(cedge["azimuth_deg"], curr_az)
            # Global heading compatibility
            if global_heading is not None:
                global_dev = abs(angle_diff(cedge["azimuth_deg"], global_heading))
                if global_dev > MAX_HEAD_DEV + 30:
                    continue  # prune completely implausible edges
            forward.append((ceid, delta))

        if not forward:
            # If all pruned, fall back to all candidates
            for ceid in candidate_eids:
                cedge = self.edge_map.get(ceid)
                if cedge is None or cedge["start_node"] != end_node:
                    continue
                delta = angle_diff(cedge["azimuth_deg"], curr_az)
                forward.append((ceid, delta))

        if not forward:
            return None

        weights = [self._turn_weight(delta, lat_accel) for _, delta in forward]
        total   = sum(weights)
        if total <= 0:
            weights = [1.0] * len(forward)
            total   = float(len(forward))
        norm_w = [w / total for w in weights]

        idx = self._weighted_choice_py(norm_w)
        return forward[idx][0]

    def _turn_weight(self, delta_deg, lat_accel):
        """
        Weight based on road direction change and lateral acceleration.

        Bearing convention:
          delta_deg > 0  ==>  RIGHT turn (clockwise increase)
          delta_deg < 0  ==>  LEFT turn

        Android accel_x with phone in cradle (X = cabin right):
          lat_accel > 0  ==>  force pushes right  ==>  LEFT turn
          lat_accel < 0  ==>  force pushes left   ==>  RIGHT turn
        """
        abs_delta = abs(delta_deg)

        if abs_delta > 150:
            return UTURN_WEIGHT

        is_turn   = abs_delta > 25
        is_strong = abs(lat_accel) >= TURN_STRONG
        is_soft   = abs(lat_accel) >= TURN_SOFT

        if not is_turn:
            # Straight-ish road (within 25 deg)
            if is_strong:   return 0.2
            elif is_soft:   return 0.8
            else:           return 5.0   # Strong straight-road bias

        road_right      = delta_deg > 0
        sensor_right    = lat_accel < 0
        direction_match = (road_right == sensor_right)

        if is_strong:
            return 4.0 if direction_match else 0.04
        elif is_soft:
            return 2.0 if direction_match else 0.2
        else:
            return 0.8   # no clear signal

    # -----------------------------------------------------------------------
    def resample(self, particles):
        """Systematic resampling with position jitter."""
        weights = np.array([p[2] for p in particles], dtype=np.float64)
        total   = weights.sum()
        if total <= 0 or not np.isfinite(total):
            weights = np.ones(len(particles)) / len(particles)
        else:
            weights /= total

        N      = self.N
        cumsum = np.cumsum(weights)
        u0     = self.np_rng.uniform(0.0, 1.0 / N)
        pos    = u0 + np.arange(N, dtype=np.float64) / N
        idxs   = np.searchsorted(cumsum, pos).clip(0, len(particles) - 1)

        new_particles = []
        for idx in idxs:
            eid, along_m, _ = particles[int(idx)]
            edge = self.edge_map.get(eid)
            if edge is None:
                new_particles.append([eid, along_m, 1.0 / N])
                continue
            jitter  = self.rng.gauss(0.0, 0.5)
            along_j = max(0.0, min(edge["length_m"], along_m + jitter))
            new_particles.append([eid, along_j, 1.0 / N])

        return new_particles

    # -----------------------------------------------------------------------
    def weighted_position(self, particles):
        """Weighted mean (lat, lon, heading_deg)."""
        if not particles:
            return None, None, None

        weights = np.array([p[2] for p in particles], dtype=np.float64)
        total   = weights.sum()
        if total <= 0:
            weights = np.ones(len(particles)) / len(particles)
        else:
            weights /= total

        lats, lons, sins, coss = [], [], [], []
        for (eid, along_m, _), w in zip(particles, weights):
            edge = self.edge_map.get(eid)
            if edge is None:
                continue
            lat, lon = point_on_edge(edge, along_m)
            az_rad   = math.radians(edge["azimuth_deg"])
            lats.append(lat * w)
            lons.append(lon * w)
            sins.append(math.sin(az_rad) * w)
            coss.append(math.cos(az_rad) * w)

        if not lats:
            return None, None, None

        mean_heading = (math.degrees(math.atan2(sum(sins), sum(coss))) + 360.0) % 360.0
        return float(sum(lats)), float(sum(lons)), mean_heading

    def unique_edges(self, particles):
        return len(set(p[0] for p in particles))

    # -----------------------------------------------------------------------
    def _point_to_edge_dist(self, plat, plon, edge):
        coords    = edge["coords"]
        best_dist = float("inf")
        best_t    = 0.0
        cum_len   = 0.0
        total_len = max(edge["length_m"], 1e-6)

        for i in range(len(coords) - 1):
            la1, lo1 = coords[i]
            la2, lo2 = coords[i + 1]
            seg_len  = haversine(la1, lo1, la2, lo2)
            d, t     = self._seg_dist(plat, plon, la1, lo1, la2, lo2)
            if d < best_dist:
                best_dist = d
                along     = cum_len + t * seg_len
                best_t    = along / total_len
            cum_len += seg_len

        return best_dist, min(1.0, best_t)

    @staticmethod
    def _seg_dist(plat, plon, lat1, lon1, lat2, lon2):
        scale  = math.cos(math.radians((lat1 + lat2) / 2)) * 111_319.5
        dx     = (lon2 - lon1) * scale
        dy     = (lat2 - lat1) * 111_319.5
        seg_sq = dx * dx + dy * dy
        if seg_sq < 1e-10:
            return haversine(plat, plon, lat1, lon1), 0.0
        px = (plon - lon1) * scale
        py = (plat - lat1) * 111_319.5
        t  = max(0.0, min(1.0, (px * dx + py * dy) / seg_sq))
        cl = lat1 + t * (lat2 - lat1)
        co = lon1 + t * (lon2 - lon1)
        return haversine(plat, plon, cl, co), t

    def _weighted_choice_py(self, weights):
        r   = self.rng.random()
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return i
        return len(weights) - 1
