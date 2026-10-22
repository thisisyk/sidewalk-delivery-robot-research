"""Observed-history urban animation using the frozen clearance-ranking model.

The classifier is a request gate, never a controller.  Buildings, trash and
barriers are not classes supported by the study.  They enter only the explicit
geometric controller and rule-based routing.  Global animation truth is kept
separate from each vehicle's recorded, building-occluded observations.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import heapq
import math

import numpy as np

from .geometry import clearance
from .simulation import state
from .urban_worlds import SCENARIO_CATALOG, make_urban_world, urban_actors


@dataclass(frozen=True)
class UrbanConfig:
    scenario: str = "intersection"
    domain: str = "road"
    threshold: float = 0.9556857111879499
    operators: int = 1
    fleet: int = 12
    seed: int = 42
    density: float = 1.
    occlusion: bool = True
    duration_s: float = 120.
    dt: float = .5
    service_s: float = 15.

    def validate(self):
        if self.scenario not in SCENARIO_CATALOG or self.domain not in ("road", "robot"):
            raise ValueError("unsupported_urban_scenario_or_domain")
        if self.operators not in (1, 2, 4) or self.fleet not in (6, 12, 24):
            raise ValueError("unsupported_urban_capacity")
        if not 0 <= self.threshold <= 1 or self.density not in (.5, 1., 1.5) or not isinstance(self.occlusion, bool):
            raise ValueError("unsupported_urban_observation_settings")
        if not 0 <= self.seed <= 100000 or not 10 <= self.duration_s <= 180 or self.dt != .5 or not 1 <= self.service_s <= 60:
            raise ValueError("urban_simulation_bounds_exceeded")


def _local(point, box):
    dx, dy = point[0] - box["x"], point[1] - box["y"]
    c, s = math.cos(box.get("heading", 0.)), math.sin(box.get("heading", 0.))
    return c * dx + s * dy, -s * dx + c * dy


def segment_hits_box(a, b, box, padding=0.):
    """Slab intersection with a rotated rectangle, using current geometry only."""
    aa, bb = _local(a, box), _local(b, box)
    lo, hi = 0., 1.
    for start, end, bound in zip(aa, bb, (box["length"] / 2 + padding, box["width"] / 2 + padding)):
        delta = end - start
        if abs(delta) < 1e-12:
            if abs(start) > bound:
                return False
            continue
        u, v = (-bound - start) / delta, (bound - start) / delta
        if u > v:
            u, v = v, u
        lo, hi = max(lo, u), min(hi, v)
        if lo > hi:
            return False
    return hi >= 0 and lo <= 1


def line_of_sight(observer, target, buildings, enabled=True):
    """Center-to-center 2D LOS.  This is a documented toy sensor, not lidar."""
    if not enabled:
        return True
    a, b = (observer["x"], observer["y"]), (target["x"], target["y"])
    return not any(segment_hits_box(a, b, box) for box in buildings)


def _near(a, b, gap=0.):
    bound = .5 * math.hypot(a["length"], a["width"]) + .5 * math.hypot(b["length"], b["width"]) + gap
    return (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 <= bound ** 2


def _gap(a, b, limit=5.):
    return clearance(a, b) if _near(a, b, limit) else limit + 1


def _body(obj):
    return state(obj["x"], obj["y"], obj.get("vx", 0.), obj.get("vy", 0.), obj["length"], obj["width"], obj.get("heading", 0.))


class ObservationHistory:
    """Eleven actually observed samples; hidden history is never back-filled."""
    def __init__(self):
        self.records = {}
        self.kinds = {}
        self.visible_ids = []
        self.hidden_ids = []

    def update(self, ego, objects, buildings, occlusion=True, radius=50.):
        current = {}
        visible, hidden = [], []
        for obj in objects:
            if obj["id"] == ego["id"] or obj.get("type") not in (1, 2, 3):
                continue
            if math.hypot(obj["x"] - ego["x"], obj["y"] - ego["y"]) > radius:
                continue
            if line_of_sight(ego, obj, buildings, occlusion):
                ident = obj["id"]
                current[ident] = _body(obj)
                self.kinds[ident] = int(obj["type"])
                visible.append(ident)
            else:
                hidden.append(obj["id"])
        for ident in set(self.records) | set(current):
            if ident not in self.records:
                self.records[ident] = deque([{"valid": False} for _ in range(10)], maxlen=11)
            self.records[ident].append(current.get(ident, {"valid": False}))
        self.visible_ids, self.hidden_ids = sorted(visible), sorted(hidden)

    def scene(self, ego, world, t):
        tracks = [{"id": ego["id"], "type": 1, "states": list(ego["history"])}]
        tracks += [{"id": ident, "type": self.kinds[ident], "states": list(self.records[ident])}
                   for ident in self.visible_ids]
        return {"current_time_index": 10, "sdc_track_index": 0,
                "timestamps": [round(t - 1 + i * .1, 8) for i in range(11)],
                "tracks": tracks, "map_features": world.get("map_features", [])}


def smooth_route(points, radius=3.):
    """Round interior route corners with sampled quadratic Bezier curves."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points.tolist()
    out = [points[0].tolist()]
    for before, corner, after in zip(points[:-2], points[1:-1], points[2:]):
        aa, bb = corner - before, after - corner
        na, nb = np.linalg.norm(aa), np.linalg.norm(bb)
        if min(na, nb) < 1e-8:
            continue
        r = min(radius, na * .25, nb * .25)
        enter, leave = corner - aa / na * r, corner + bb / nb * r
        out.append(enter.tolist())
        for u in np.linspace(.125, 1., 8):
            out.append(((1 - u) ** 2 * enter + 2 * u * (1 - u) * corner + u ** 2 * leave).tolist())
    out.append(points[-1].tolist())
    return out


class Route:
    def __init__(self, points, smooth=True):
        raw = smooth_route(points) if smooth else points
        cleaned = [list(map(float, raw[0]))]
        for point in raw[1:]:
            if math.dist(point, cleaned[-1]) > 1e-8:
                cleaned.append(list(map(float, point)))
        self.points = cleaned
        self.lengths = [math.dist(a, b) for a, b in zip(cleaned[:-1], cleaned[1:])]
        self.cumulative = np.concatenate(([0.], np.cumsum(self.lengths)))
        self.total = float(self.cumulative[-1])

    def pose(self, distance):
        if not self.lengths:
            return *self.points[0], 0.
        s = min(max(0., distance), self.total)
        i = min(len(self.lengths) - 1, int(np.searchsorted(self.cumulative, s, side="right") - 1))
        a, b = self.points[i], self.points[i + 1]
        u = (s - self.cumulative[i]) / self.lengths[i]
        return a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), math.atan2(b[1] - a[1], b[0] - a[0])


def _point_segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    den = dx * dx + dy * dy
    u = max(0., min(1., ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / den)) if den else 0.
    return math.hypot(p[0] - a[0] - u * dx, p[1] - a[1] - u * dy)


def _point_box_gap(p, box):
    x, y = _local(p, box)
    return math.hypot(max(0., abs(x) - box["length"] / 2), max(0., abs(y) - box["width"] / 2))


def plan_static_detour(vehicle, world, known_obstacles, grid=2.):
    """Rule-based A* on mapped roads/buildings plus previously observed objects.

    A conservative circumscribed footprint is checked along every edge.  Hidden
    trash/barriers are not inputs.  Dynamic traffic is handled by the local
    visible-object guard; no learned path planning or legal lane claim is made.
    """
    radius = .5 * math.hypot(vehicle["length"], vehicle["width"]) + .3
    blocks = list(world.get("buildings", [])) + list(known_obstacles)
    start = (vehicle["x"], vehicle["y"])
    goal = tuple(vehicle["goal"])
    edges = [(a, b, road["width"] / 2 - radius) for road in world["roads"]
             for a, b in zip(road["points"][:-1], road["points"][1:])]
    def free(point):
        return (0 <= point[0] <= world["width"] and 0 <= point[1] <= world["height"]
                and any(_point_segment_distance(point, a, b) <= width for a, b, width in edges)
                and all(_point_box_gap(point, box) > radius for box in blocks))
    def edge_free(a, b):
        n = max(1, int(math.ceil(math.dist(a, b) / .4)))
        return all(free((a[0] + (b[0] - a[0]) * j / n, a[1] + (b[1] - a[1]) * j / n)) for j in range(n + 1))
    # The root state need not lie exactly on the grid.  Nearby nodes are only
    # admitted if the entire attachment is collision free.
    candidates = [(x * grid, y * grid) for x in range(max(0, int(start[0] / grid) - 2), int(start[0] / grid) + 3)
                  for y in range(max(0, int(start[1] / grid) - 2), int(start[1] / grid) + 3)]
    initial = [p for p in candidates if edge_free(start, p)]
    if not initial or not free(goal):
        return None
    frontier, costs, parents = [], {}, {}
    for p in initial:
        cost = math.dist(start, p)
        costs[p] = cost; parents[p] = None
        heapq.heappush(frontier, (cost + math.dist(p, goal), p))
    found = None
    while frontier:
        _, p = heapq.heappop(frontier)
        if math.dist(p, goal) <= grid * 2 and edge_free(p, goal):
            found = p; break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            q = (p[0] + dx * grid, p[1] + dy * grid)
            cost = costs[p] + grid * math.hypot(dx, dy)
            if cost < costs.get(q, float("inf")) and edge_free(p, q):
                costs[q] = cost; parents[q] = p
                heapq.heappush(frontier, (cost + math.dist(q, goal), q))
    if found is None:
        return None
    reverse = [goal, found]
    while parents[found] is not None:
        found = parents[found]; reverse.append(found)
    points = [start] + list(reversed(reverse))
    # Greedy line-of-sight shortening preserves the same footprint checks.
    simplified, index = [points[0]], 0
    while index < len(points) - 1:
        end = len(points) - 1
        while end > index + 1 and not edge_free(points[index], points[end]):
            end -= 1
        simplified.append(points[end]); index = end
    return Route(simplified, smooth=False)


def _public_vehicle(v):
    keys = ("id", "x", "y", "vx", "vy", "heading", "length", "width", "status", "score", "eligible",
            "requested", "done", "route_index", "route_progress", "visible_count", "hidden_count",
            "visible_actor_ids", "hidden_actor_ids", "unsupported_obstacles", "route_replans", "replan_failed")
    return {k: v[k] for k in keys}


def _run_urban(model, cfg, policy="learned", capture=True):
    cfg.validate()
    world = make_urban_world(cfg.scenario, domain=cfg.domain)
    rng = np.random.default_rng(cfg.seed)
    length, width, speed = (4.5, 1.8, 3.) if cfg.domain == "road" else (.9, .65, 1.3)
    vehicles, observers = [], []
    routes = [Route(points) for points in world["routes"]]
    for i in range(cfg.fleet):
        route_index = i % len(routes); route = routes[route_index]
        x, y, heading = route.pose(0.)
        body = state(x, y, length=length, width=width, heading=heading)
        vehicles.append({"id": i, **body, "route": route, "route_index": route_index, "goal": route.points[-1],
            "route_progress": 0., "start": (i // len(routes)) * 8., "speed": speed * rng.uniform(.92, 1.08),
            "active": False, "done": False, "history": deque([dict(body) for _ in range(11)], maxlen=11),
            "score": None, "eligible": False, "blocked_s": 0., "wait_s": 0., "status": "pending",
            "requested": False, "served": False, "request_time": None, "route_replans": 0, "replan_failed": False,
            "known_obstacles": {}, "visible_count": 0, "hidden_count": 0, "visible_actor_ids": [], "hidden_actor_ids": [],
            "unsupported_obstacles": [], "fresh_score_t": None})
        observers.append(ObservationHistory())
    operators = [{"id": i, "vehicle_id": None, "remaining": 0.} for i in range(cfg.operators)]
    queue, frames, events, score_series = [], [], [], []
    requests = occupied_s = score_count = out_of_range = 0
    near_frames = overlap_frames = occluded_frames = 0
    contact_by_kind = {}
    stationary_contact_frames = 0
    supported_static = [obj for obj in world.get("obstacles", []) if obj.get("kind") == "parked_car"]
    unsupported = [obj for obj in world.get("obstacles", []) if obj.get("kind") != "parked_car"]
    def event(t, v, message):
        if capture:
            events.append({"t": round(t, 2), "id": v["id"], "message": message})
    physics_dt = .1
    steps = int(round(cfg.duration_s / physics_dt))
    for tick in range(steps):
        t = round(tick * physics_dt, 8)
        actors = urban_actors(t, cfg.scenario, seed=cfg.seed, density=cfg.density)
        for v in vehicles:
            if not v["active"] and t >= v["start"] and all(not o["active"] or o["done"] or o["id"] == v["id"] or _gap(v, o, 3.) >= 3. for o in vehicles):
                v["active"] = True; v["status"] = "moving"
        traffic = [{**_body(o), "id": o["id"], "type": 1} for o in vehicles if o["active"] and not o["done"]]
        traffic += actors + [{**obj, "type": 1} for obj in supported_static]
        visible_objects = {}
        for v, obs in zip(vehicles, observers):
            if not v["active"] or v["done"]:
                continue
            obs.update(v, traffic, world.get("buildings", []), cfg.occlusion)
            v["visible_actor_ids"] = list(obs.visible_ids); v["hidden_actor_ids"] = list(obs.hidden_ids)
            v["visible_count"] = len(obs.visible_ids); v["hidden_count"] = len(obs.hidden_ids)
            visible_objects[v["id"]] = [o for o in traffic if o["id"] in obs.visible_ids]
            for obj in world.get("obstacles", []):
                if math.hypot(obj["x"] - v["x"], obj["y"] - v["y"]) <= 35 and line_of_sight(v, obj, world.get("buildings", []), cfg.occlusion):
                    v["known_obstacles"][obj["id"]] = obj
            v["unsupported_obstacles"] = [ident for ident, obj in v["known_obstacles"].items() if obj.get("kind") != "parked_car"]
            # Do not display a stale score as applicable after the current
            # observation enters the study's excluded <=2 m region.
            if not obs.visible_ids or any(_gap(v, obj, 2.) <= 2. for obj in visible_objects[v["id"]]):
                v["score"] = None; v["eligible"] = False
        if tick % 10 == 0 and (policy == "learned" or capture):
            targets, rows = [], []
            for v, obs in zip(vehicles, observers):
                if not v["active"] or v["done"]:
                    continue
                try:
                    rows.append(model.features_from_scene(obs.scene(v, world, t))); targets.append(v)
                except ValueError as exc:
                    if str(exc) != "no_current_neighbors":
                        raise
                    v["score"] = None; v["eligible"] = False
            predictions = model.predict_features_many(rows) if rows else []
            for v, prediction in zip(targets, predictions):
                v["eligible"] = bool(prediction["score_applicable"])
                v["score"] = float(prediction["score"]) if v["eligible"] else None
                v["fresh_score_t"] = t
                score_count += 1; out_of_range += bool(prediction["outside_training_range_features"])
        for operator in operators:
            if operator["vehicle_id"] is None:
                continue
            occupied_s += physics_dt
            operator["remaining"] = max(0., operator["remaining"] - physics_dt)
            if operator["remaining"] <= 1e-8:
                v = vehicles[operator["vehicle_id"]]
                v["served"] = True
                route = plan_static_detour(v, world, list(v["known_obstacles"].values()))
                if route is None:
                    v["replan_failed"] = True
                    event(t, v, "规则路径重规划未找到可通行路线；协助不保证完成")
                else:
                    v["route"] = route; v["route_progress"] = 0.; v["route_replans"] += 1
                    event(t, v, "模拟协助完成，尝试已知静态障碍上的规则 A* 路线")
                operator["vehicle_id"] = None
        # Proposals use the snapshot above, independent of update order.
        proposals = []
        for v in vehicles:
            if not v["active"] or v["done"]:
                continue
            busy = any(o["vehicle_id"] == v["id"] for o in operators)
            progress = min(v["route"].total, v["route_progress"] + v["speed"] * physics_dt)
            x, y, heading = v["route"].pose(progress)
            trial = state(x, y, length=length, width=width, heading=heading)
            # Only current visible motion is linearly extrapolated by 0.1 s.
            guards = [{**o, "x": o["x"] + o.get("vx", 0.) * physics_dt,
                       "y": o["y"] + o.get("vy", 0.) * physics_dt} for o in visible_objects[v["id"]]]
            static = list(world.get("buildings", [])) + list(v["known_obstacles"].values())
            dynamic_margin = 2.25 if cfg.domain == "road" else .55
            static_margin = 1.1 if cfg.domain == "road" else .35
            safe = (all(_gap(trial, obj, dynamic_margin) >= dynamic_margin for obj in guards)
                    and all(_gap(trial, obj, static_margin) >= static_margin for obj in static))
            # Rotation along short route corners is checked as well.
            for u in (.5,):
                if not safe:
                    break
                xx, yy, hh = v["route"].pose(v["route_progress"] + (progress - v["route_progress"]) * u)
                middle = state(xx, yy, length=length, width=width, heading=hh)
                safe = (all(_gap(middle, obj, dynamic_margin) >= dynamic_margin for obj in guards)
                        and all(_gap(middle, obj, static_margin) >= static_margin for obj in static))
            blocked = not safe or busy or v["id"] in queue
            if blocked:
                v["blocked_s"] += physics_dt; v["wait_s"] += physics_dt
                proposals.append((v, v["x"], v["y"], v["heading"], v["route_progress"], True, busy))
            else:
                v["blocked_s"] = 0.
                proposals.append((v, x, y, heading, progress, False, busy))
        for v, x, y, heading, progress, blocked, busy in proposals:
            v["vx"], v["vy"] = (x - v["x"]) / physics_dt, (y - v["y"]) / physics_dt
            v["x"], v["y"], v["heading"], v["route_progress"] = x, y, heading, progress
            if progress >= v["route"].total - 1e-6:
                v["done"] = True; v["status"] = "done"; v["score"] = None; v["eligible"] = False
                v["vx"] = v["vy"] = 0.
                event(t + physics_dt, v, "到达模拟路线终点")
            elif busy:
                v["status"] = "assisting"
            elif v["id"] in queue:
                v["status"] = "queued"
            elif blocked:
                v["status"] = "waiting"
            else:
                v["status"] = "bypassing" if v["route_replans"] else "moving"
            if v["blocked_s"] >= 2 - 1e-8 and not v["requested"] and not v["done"]:
                request = policy == "all" or (policy == "learned" and v["eligible"] and v["score"] is not None and v["score"] >= cfg.threshold)
                if request:
                    requests += 1; v["requested"] = True; v["request_time"] = t; queue.append(v["id"])
                    event(t, v, "真实学习模型分数达到阈值，请求模拟协助" if policy == "learned" else "等待规则请求模拟协助")
        for v in vehicles:
            v["history"].append(_body(v))
        for operator in operators:
            if operator["vehicle_id"] is None and queue:
                ident = queue.pop(0); v = vehicles[ident]
                operator.update(vehicle_id=ident, remaining=cfg.service_s)
                v["status"] = "assisting"; event(t, v, "模拟操作员开始处理")
        if (tick + 1) % 5 == 0:
            # Truth is used exclusively for evaluation/rendering, never features.
            truth_actors = urban_actors(round(t + physics_dt, 8), cfg.scenario, seed=cfg.seed, density=cfg.density)
            active = [v for v in vehicles if v["active"] and not v["done"]]
            truth_static = world.get("buildings", []) + world.get("obstacles", [])
            contact_ids, contact_pairs = [], []
            for v in active:
                truth = [o for o in active if o["id"] != v["id"]] + truth_actors + truth_static
                near_frames += any(_gap(v, obj, .1) <= .1 for obj in truth)
                # At <=1e-8, contact and overlap are not distinguished by the
                # study's nonnegative clearance; retain the conservative name.
                contacts = [obj for obj in truth if _gap(v, obj, .000001) <= 1e-8]
                overlap_frames += bool(contacts)
                if contacts:
                    contact_ids.append(v["id"])
                    stationary_contact_frames += math.hypot(v["vx"], v["vy"]) < .05
                    kinds = set()
                    for obj in contacts:
                        kind = obj.get("kind") or ({1: "vehicle", 2: "pedestrian", 3: "cyclist"}.get(obj.get("type"), "fleet_vehicle"))
                        kinds.add(kind)
                        contact_pairs.append({"vehicle_id": v["id"], "object_id": obj["id"], "kind": kind,
                            "vehicle_stationary": math.hypot(v["vx"], v["vy"]) < .05})
                    for kind in kinds:
                        contact_by_kind[kind] = contact_by_kind.get(kind, 0) + 1
                occluded_frames += v["hidden_count"]
            if capture:
                main = vehicles[0]
                frames.append({"t": round(t + physics_dt, 2), "vehicles": [_public_vehicle(v) for v in vehicles if v["active"]],
                    "actors": truth_actors, "queue": [{"id": i, "score": vehicles[i]["score"]} for i in queue],
                    "operators": [dict(o) for o in operators], "completed": sum(v["done"] for v in vehicles),
                    "visible_actor_ids": list(main["visible_actor_ids"]), "hidden_actor_ids": list(main["hidden_actor_ids"]),
                    "contact_vehicle_ids": contact_ids, "contact_pairs": contact_pairs})
                score_series.append({"t": round(t + physics_dt, 2), "score": main["score"], "threshold": cfg.threshold, "eligible": main["eligible"]})
    waits = [v["wait_s"] for v in vehicles]
    summary = {"completed": sum(v["done"] for v in vehicles), "total": cfg.fleet, "unfinished": sum(not v["done"] for v in vehicles),
        "mean_wait_s": round(float(np.mean(waits)), 2), "p95_wait_s": round(float(np.percentile(waits, 95)), 2),
        "requests": int(requests), "operator_busy_fraction": min(1., occupied_s / (cfg.operators * cfg.duration_s)),
        "near_contact_frames": int(near_frames), "physical_overlap_frames": int(overlap_frames),
        "contact_vehicle_frames_by_object_kind": contact_by_kind,
        "contact_stationary_vehicle_frames": int(stationary_contact_frames),
        "occluded_actor_frames": int(occluded_frames), "unsupported_static_objects": len(unsupported) + len(world.get("buildings", [])),
        "waiting_at_end": sum(v["active"] and not v["done"] and v["status"] in ("waiting", "assisting", "queued") for v in vehicles),
        "route_replans": sum(v["route_replans"] for v in vehicles), "failed_replans": sum(v["replan_failed"] for v in vehicles),
        "fraction_scored_frames_outside_training_minmax": out_of_range / score_count if score_count else None,
        "domain_warning": "合成道路、遮挡观测及控制均有分布偏移；模型未学习建筑/垃圾/围挡" + ("；机器人尺寸/速度属训练域外" if cfg.domain == "robot" else ""),
        "policy": policy}
    return {"config": asdict(cfg), "frames": frames, "events": events, "score_series": score_series, "world": world, "summary": summary,
        "metadata": {"synthetic": True, "model_scores": "actual_frozen_hist_gradient_boosting_predict_proba",
            "score_role": "request_gate_only; uncalibrated clearance proxy, not danger probability",
            "observation": "actual 0.1-second per-ego history; building centerline LOS; no hidden-history backfill",
            "score_interval_s": 1., "physics_interval_s": .1, "render_interval_s": .5,
            "supported_model_objects": ["vehicle", "pedestrian", "cyclist"],
            "unsupported_model_objects": ["building", "trash", "barrier"],
            "static_map_assumption": "building/road map known; other static obstacles recorded only after LOS observation within35m",
            "routing": "rounded supplied route; after service conservative static-grid A* using known obstacles; can fail",
            "safety_guard": "independent visible-dynamic clearance>=2.25m road/0.55m robot and static clearance>=1.1m road/0.35m robot; no hidden dynamic truth",
            "overlap_metric": "number of 0.5s vehicle frames with true 2D rectangle contact or overlap; counted per vehicle, not unique collisions",
            "contact_kind_metric": "per object-kind vehicle-frame counts; multi-kind contacts can count in multiple kinds",
            "occlusion_metric": "sum of hidden supported actor counts across active vehicle frames at0.5s",
            "one_request_per_vehicle": True, "queue": "FIFO", "service_duration_s": cfg.service_s,
            "not_validated": ["safety", "traffic rules", "assistance need", "delivery cost", "sensor realism", "probability calibration"]}}


def simulate_urban(model, config=None, compare=True):
    cfg = config if isinstance(config, UrbanConfig) else UrbanConfig(**(config or {}))
    result = _run_urban(model, cfg, "learned", True)
    if compare:
        summaries = [result["summary"]] + [_run_urban(model, cfg, p, False)["summary"] for p in ("all", "no_assist")]
        names = {"learned": "学习模型门控", "all": "等待2秒即请求", "no_assist": "仅自主等待"}
        result["policies"] = [{"name": names[row["policy"]], **row} for row in summaries]
    return result
