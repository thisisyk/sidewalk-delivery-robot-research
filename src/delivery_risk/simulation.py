"""Synthetic animation world driven by the frozen learned model's real scores.

The learned model ranks three-second near-clearance proxy events. It is NOT a
validated controller or assistance-demand classifier. Motion guards, routing,
service duration and assistance outcomes below are explicit toy assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from .geometry import clearance


@dataclass(frozen=True)
class SimulationConfig:
    scenario: str = "blocked"
    domain: str = "road"
    threshold: float = 0.9556857111879499
    operators: int = 1
    fleet: int = 12
    seed: int = 42
    duration_s: float = 120.
    dt: float = .5
    service_s: float = 15.

    def validate(self):
        if self.scenario not in ("crossing", "blocked", "congestion") or self.domain not in ("road", "robot"):
            raise ValueError("unsupported_scenario_or_domain")
        if not 0 <= self.threshold <= 1 or self.operators not in (1, 2, 4) or self.fleet not in (6, 12, 24):
            raise ValueError("unsupported_simulation_settings")
        if not 0 <= self.seed <= 100000 or not 10 <= self.duration_s <= 180 or self.dt != .5 or not 1 <= self.service_s <= 60:
            raise ValueError("simulation_bounds_exceeded")


def state(x, y, vx=0., vy=0., length=4.5, width=1.8, heading=0.):
    return {"x": float(x), "y": float(y), "vx": float(vx), "vy": float(vy),
            "length": length, "width": width, "heading": heading, "valid": True}


def make_world(cfg):
    obstacles = []
    if cfg.scenario in ("blocked", "congestion"):
        obstacles.append({"id": 900, **state(55., 20., length=5., width=2.)})
    if cfg.scenario == "congestion":
        obstacles.append({"id": 901, **state(66., 8., length=5., width=2.)})
    return {"width": 100, "height": 40, "goal_x": 92, "lanes": [8, 20, 32], "obstacles": obstacles}


def moving_actors(t, cfg):
    actors = []
    if cfg.scenario in ("crossing", "congestion"):
        # Scripted exogenous actors; their future is never passed to the model.
        phase = (t + (cfg.seed % 7)) % 34.
        if phase <= 24:
            actors.append({"id": 800, "type": 2,
                **state(44., 3. + phase * 1.4, vy=1.4, length=.6, width=.6, heading=math.pi/2)})
        if cfg.scenario == "congestion":
            phase2 = (t + 13) % 39.
            if phase2 <= 27:
                actors.append({"id": 801, "type": 3,
                    **state(74., 38. - phase2 * 1.2, vy=-1.2, length=1.7, width=.6, heading=-math.pi/2)})
    return actors


def _scene(vehicle, vehicles, actors, world, t):
    # Construct exactly eleven past/current states, with no future trajectories.
    times = [t - 1. + i*.1 for i in range(11)]
    tracks = [{"id": vehicle["id"], "type": 1, "states": vehicle["history"][-11:]}]
    for other in vehicles:
        if other["id"] != vehicle["id"] and other["active"] and not other["done"]:
            tracks.append({"id": other["id"], "type": 1, "states": other["history"][-11:]})
    for actor in actors:
        history = [state(actor["x"] - actor["vx"]*(t-ti), actor["y"] - actor["vy"]*(t-ti),
            actor["vx"], actor["vy"], actor["length"], actor["width"], actor["heading"]) for ti in times]
        tracks.append({"id": actor["id"], "type": actor["type"], "states": history})
    for obstacle in world["obstacles"]:
        tracks.append({"id": obstacle["id"], "type": 1, "states": [dict(obstacle) for _ in times]})
    maps = [{"type": "lane", "polyline": [[x, lane] for x in range(0, 101, 5)]} for lane in world["lanes"]]
    maps += [{"type": "road_edge", "polyline": [[x, y] for x in range(0, 101, 5)]} for y in (2., 38.)]
    maps += [{"type": "crosswalk", "polygon": [[42, 2], [46, 2], [46, 38], [42, 38]]}]
    return {"current_time_index": 10, "timestamps": times, "sdc_track_index": 0, "tracks": tracks, "map_features": maps}


def _run(model, cfg, policy="learned", capture=True):
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    world = make_world(cfg)
    length, width, speed = (4.5, 1.8, 3.) if cfg.domain == "road" else (.9, .65, 1.3)
    vehicles = []
    for i in range(cfg.fleet):
        lane = [20., 8., 32.][i % 3]
        s = state(5., lane, length=length, width=width)
        vehicles.append({"id": i, **s, "lane": lane, "start": (i//3)*12., "speed": speed * rng.uniform(.92, 1.08),
            "active": False, "done": False, "history": [dict(s) for _ in range(11)], "score": None, "eligible": False,
            "blocked_s": 0., "wait_s": 0., "status": "pending", "requested": False, "served": False,
            "bypass": False, "phase": "out", "bypass_end": 0., "request_time": None, "queue_wait": None})
    operators = [{"id": i, "vehicle_id": None, "remaining": 0.} for i in range(cfg.operators)]
    queue, frames, events, score_series = [], [], [], []
    requests = 0; occupied_s = 0.; near_frames = 0; score_count = 0; out_of_range = 0
    def event(t, v, message):
        if capture:
            events.append({"t": round(t, 2), "id": v["id"], "message": message})
    steps = int(cfg.duration_s/cfg.dt)
    for step in range(steps):
        t = step*cfg.dt
        actors = moving_actors(t, cfg)
        for v in vehicles:
            if not v["active"] and t >= v["start"]:
                # Defer spawn if a previous vehicle still occupies its entry.
                if all(not o["active"] or o["done"] or o["id"] == v["id"] or clearance(v, o) >= 3 for o in vehicles):
                    v["active"] = True; v["status"] = "moving"
        # All policies score with the same real model in the visualized run;
        # non-visual baselines need no unnecessary inference.
        if step % 2 == 0 and (policy == "learned" or capture):
            score_vehicles, feature_rows = [], []
            for v in vehicles:
                if not v["active"] or v["done"]:
                    continue
                try:
                    feature_rows.append(model.features_from_scene(_scene(v, vehicles, actors, world, t)))
                    score_vehicles.append(v)
                except ValueError as exc:
                    if str(exc) != "no_current_neighbors":
                        raise
                    v["score"] = None; v["eligible"] = False
            predictions = model.predict_features_many(feature_rows) if feature_rows else []
            for v, prediction in zip(score_vehicles, predictions):
                v["score"] = prediction["score"]
                v["eligible"] = prediction["score_applicable"]
                score_count += 1
                out_of_range += bool(prediction["outside_training_range_features"])
        for operator in operators:
            if operator["vehicle_id"] is None:
                continue
            occupied_s += cfg.dt
            operator["remaining"] = max(0., operator["remaining"] - cfg.dt)
            if operator["remaining"] <= 0:
                v = vehicles[operator["vehicle_id"]]
                v["served"] = True; v["bypass"] = True; v["phase"] = "out"
                ahead = [o["x"]+o["length"]/2 + length/2 + 4 for o in world["obstacles"] if abs(o["y"]-v["lane"]) < 3 and o["x"] > v["x"]]
                v["bypass_end"] = max(ahead, default=v["x"]+12.)
                event(t, v, "模拟协助完成，按预设绕行路线尝试前进")
                operator["vehicle_id"] = None
        for v in vehicles:
            if not v["active"] or v["done"]:
                continue
            busy = any(o["vehicle_id"] == v["id"] for o in operators)
            dx = v["speed"]*cfg.dt; dy = 0.
            if v["bypass"]:
                target = v["lane"]+4.3
                if v["phase"] == "out":
                    dy = min(1.2*cfg.dt, target-v["y"]); dx = 0.
                    if target-v["y"] < .01:
                        v["phase"] = "past"
                elif v["phase"] == "past" and v["x"] > v["bypass_end"]:
                    v["phase"] = "back"
                elif v["phase"] == "back":
                    dy = max(-1.2*cfg.dt, v["lane"]-v["y"]); dx = 0.
                    if abs(v["lane"]-v["y"]) < .01:
                        v["bypass"] = False
            trial = {**v, "x": v["x"]+dx, "y": v["y"]+dy}
            others = [o for o in vehicles if o["active"] and not o["done"] and o["id"] != v["id"]]
            guards = others + world["obstacles"] + actors
            # Independent toy guard, not learned vehicle control. Orientation is
            # fixed in this simple lane/side-shift world and explicitly visible.
            safe = all(clearance(trial, o) >= 2.25 for o in guards)
            blocked = not safe or busy
            if blocked:
                dx = dy = 0.
                v["blocked_s"] += cfg.dt; v["wait_s"] += cfg.dt
            else:
                v["blocked_s"] = 0.
            v["x"] += dx; v["y"] += dy; v["vx"] = dx/cfg.dt; v["vy"] = dy/cfg.dt
            if v["x"] >= world["goal_x"]:
                v["done"] = True; v["status"] = "done"; v["score"] = None; v["eligible"] = False
                event(t+cfg.dt, v, "到达模拟终点")
            elif busy:
                v["status"] = "assisting"
            elif v["id"] in queue:
                v["status"] = "queued"
            elif blocked:
                v["status"] = "waiting"
            else:
                v["status"] = "bypassing" if v["bypass"] else "moving"
            if v["blocked_s"] >= 2 and not v["requested"] and not v["done"]:
                should_request = policy == "all" or (policy == "learned" and v["eligible"] and v["score"] is not None and v["score"] >= cfg.threshold)
                if should_request:
                    requests += 1; v["requested"] = True; v["request_time"] = t; queue.append(v["id"])
                    event(t, v, "模型分数达到阈值，请求模拟协助" if policy == "learned" else "固定规则请求模拟协助")
            if guards and min(clearance(v, o) for o in guards) <= .1:
                near_frames += 1
            # Interpolate the last half-second to provide actual 0.1s history.
            for j in range(1, 6):
                v["history"].append(state(v["x"]-dx*(1-j/5), v["y"]-dy*(1-j/5), v["vx"], v["vy"], length, width))
            v["history"] = v["history"][-11:]
        # FIFO is shared across strategies, no future labels choose service.
        for operator in operators:
            if operator["vehicle_id"] is None and queue:
                ident = queue.pop(0); v = vehicles[ident]
                v["queue_wait"] = t-v["request_time"]
                operator.update(vehicle_id=ident, remaining=cfg.service_s)
                v["status"] = "assisting"; event(t, v, "模拟操作员开始处理")
        completed = sum(v["done"] for v in vehicles)
        if capture:
            frames.append({"t": t+cfg.dt, "vehicles": [{k: v[k] for k in ("id", "x", "y", "heading", "length", "width", "status", "score", "eligible", "requested", "done")} for v in vehicles if v["active"]],
                "actors": actors, "queue": [{"id": i, "score": vehicles[i]["score"]} for i in queue],
                "operators": [dict(o) for o in operators], "completed": completed})
            main = vehicles[0]
            score_series.append({"t": t+cfg.dt, "score": main["score"], "threshold": cfg.threshold, "eligible": main["eligible"]})
    waits = [v["wait_s"] for v in vehicles]
    warning = "人行道机器人尺寸/速度属于训练域外；分数未经此域校准" if cfg.domain == "robot" else "道路车尺寸接近训练对象，但合成运动/地图仍有分布偏移"
    summary = {"completed": sum(v["done"] for v in vehicles), "total": cfg.fleet,
        "mean_wait_s": round(float(np.mean(waits)), 2), "p95_wait_s": round(float(np.percentile(waits, 95)), 2),
        "requests": requests, "operator_busy_fraction": min(1., occupied_s/(cfg.operators*cfg.duration_s)),
        "near_contact_frames": near_frames, "domain_warning": warning,
        "fraction_scored_frames_outside_training_minmax": out_of_range/score_count if score_count else None,
        "unfinished": cfg.fleet - sum(v["done"] for v in vehicles), "policy": policy}
    return {"config": asdict(cfg), "summary": summary, "frames": frames, "events": events, "score_series": score_series, "world": world,
        "metadata": {"synthetic": True, "model_scores": "actual_frozen_hist_gradient_boosting_predict_proba",
            "score_role": "request_gate_only; not vehicle controller", "score_interval_s": 1,
            "one_request_per_vehicle": True, "queue": "FIFO", "service_duration_s": cfg.service_s,
            "routing": "scripted_4.3m_side_shift_after_service", "safety_guard": "next-step 2D clearance >=2.25m; no continuous collision guarantee",
            "not_validated": ["assistance need", "probability calibration", "delivery time", "safety", "cost", "controller"]}}


def simulate(model, config=None, compare=True):
    cfg = config if isinstance(config, SimulationConfig) else SimulationConfig(**(config or {}))
    result = _run(model, cfg, "learned", True)
    if compare:
        summaries = [result["summary"]]
        summaries += [_run(model, cfg, policy, False)["summary"] for policy in ("all", "no_assist")]
        names = {"learned": "学习模型门控", "all": "等待2秒即请求", "no_assist": "仅自主等待"}
        result["policies"] = [{"name": names[row["policy"]], **row} for row in summaries]
    return result
