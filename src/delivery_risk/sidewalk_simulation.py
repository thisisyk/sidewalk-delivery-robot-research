"""Sidewalk delivery robots, separated from road traffic by a hard motion domain.

The frozen road-vehicle HistGB supplies only an uncalibrated request-gate score.
Motion, yielding, geometric clearance and A* remain explicit simulator rules.
No robot footprint may enter a motor road except an actual crosswalk polygon.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import heapq
import math

import numpy as np

from .simulation import state
from .urban_simulation import (ObservationHistory, Route, _body, _gap, _point_box_gap,
                               _public_vehicle, line_of_sight)
from .sidewalk_worlds import (SIDEWALK_CATALOG, make_sidewalk_world, sidewalk_actors,
    point_in_walkable, footprint_in_walkable, footprint_in_road)
from .assistance import (ASSISTANCE_VERSION, ASSISTANCE_ACTIONS, ACTION_LABELS,
    plan_assistance, new_incident, record_incident_state)


FALLBACK_RULES = {
    "persistent_stall_s": 5., "stationary_robot_blockage_s": 2.,
    "consecutive_planning_failures": 2, "retry_cooldown_after_service_s": 15.,
    "max_requests_per_robot": 3, "waiting_queue_slots_per_operator": 2,
    "minimum_fleet_request_interval_s": .5,
}


def uses_stall_fallback(policy, cfg):
    return policy in ("learned_with_fallback", "fallback_only") and cfg.fallback_enabled


@dataclass(frozen=True)
class SidewalkConfig:
    scenario: str = "intersection"
    domain: str = "robot"
    threshold: float = 0.9556857111879499
    operators: int = 1
    fleet: int = 12
    seed: int = 42
    density: float = 1.
    occlusion: bool = True
    fallback_enabled: bool = True
    assistance_module_enabled: bool = True
    assistance_action: str = "auto"
    duration_s: float = 120.
    dt: float = .5
    service_s: float = 15.

    def validate(self):
        if self.domain != "robot":
            raise ValueError("sidewalk_demo_requires_delivery_robot_domain")
        if self.scenario not in SIDEWALK_CATALOG:
            raise ValueError("unsupported_sidewalk_scenario")
        if self.fleet not in (6, 12, 24) or self.operators not in (1, 2, 4):
            raise ValueError("unsupported_sidewalk_capacity")
        if self.density not in (.5, 1., 1.5) or not isinstance(self.occlusion, bool) or not isinstance(self.fallback_enabled, bool) or not 0 <= self.threshold <= 1:
            raise ValueError("unsupported_sidewalk_observation_settings")
        if not 0 <= self.seed <= 100000 or not 10 <= self.duration_s <= 180 or self.dt != .5 or not 1 <= self.service_s <= 60:
            raise ValueError("sidewalk_simulation_bounds_exceeded")
        if not isinstance(self.assistance_module_enabled, bool) or self.assistance_action not in ASSISTANCE_ACTIONS:
            raise ValueError("unsupported_assistance_command_settings")


def assistance_request_reason(robot, policy, cfg, now, pending=False):
    """Score-independent stall fallback; queue/service never count as new stalls.

    These fixed thresholds are engineering assumptions, not learned parameters.
    Eligibility and the original model score are never fabricated for fallback.
    """
    if pending or robot["done"] or policy == "no_assist":
        return None
    hybrid = uses_stall_fallback(policy, cfg)
    if not hybrid:
        if robot["requested"] or robot["blocked_s"] < 2 - 1e-8:
            return None
        if policy == "all":
            return "legacy_wait_rule"
        if policy in ("learned", "learned_with_fallback") and robot["eligible"] and robot["score"] is not None and robot["score"] >= cfg.threshold:
            return "model_threshold"
        return None
    if robot["request_count"] >= FALLBACK_RULES["max_requests_per_robot"] or now + 1e-8 < robot["next_request_allowed_t"]:
        return None
    if robot["stalled_s"] >= 2 - 1e-8 and robot["eligible"] and robot["score"] is not None and robot["score"] >= cfg.threshold:
        return "model_threshold"
    if robot["stalled_s"] > 0 and robot["consecutive_planning_failures"] >= FALLBACK_RULES["consecutive_planning_failures"]:
        return "repeated_planning_failure"
    if robot["robot_blockage_s"] >= FALLBACK_RULES["stationary_robot_blockage_s"] - 1e-8:
        return "robot_blockage"
    if robot["stalled_s"] >= FALLBACK_RULES["persistent_stall_s"] - 1e-8:
        return "persistent_stall"
    return None


def plan_sidewalk_detour(robot, world, known_obstacles, grid=1.):
    """Conservative static A* in walkway/crosswalk union; motor roads forbidden.

    A circumscribed disk supports arbitrary turn headings.  Edge sampling also
    checks a full footprint at the proposed edge heading.  Buildings are mapped;
    trash/barriers/parked cars become known only after actual LOS observation.
    """
    radius = math.hypot(robot["length"], robot["width"]) / 2 + .12 + .125
    blocks = list(world.get("buildings", [])) + list(known_obstacles)
    start, goal = (robot["x"], robot["y"]), tuple(robot["goal"])
    cache = {}
    def free(p):
        key = (round(p[0], 6), round(p[1], 6))
        if key not in cache:
            # An axis-aligned square containing the circumscribed disk proves
            # containment for every robot heading, including node rotations.
            # Extra .125m covers the maximum unsampled translation between
            # adjacent .25m edge checks.
            cache[key] = (point_in_walkable(p, world)
                and footprint_in_walkable(state(*p, length=2 * radius, width=2 * radius), world)
                and all(_point_box_gap(p, obj) > radius for obj in blocks))
        return cache[key]
    def edge_free(a, b):
        count = max(1, math.ceil(math.dist(a, b) / .25))
        for i in range(count + 1):
            p = (a[0] + (b[0] - a[0]) * i / count, a[1] + (b[1] - a[1]) * i / count)
            if not free(p):
                return False
        return True
    candidates = [(x * grid, y * grid)
        for x in range(max(0, int(start[0] / grid) - 2), int(start[0] / grid) + 3)
        for y in range(max(0, int(start[1] / grid) - 2), int(start[1] / grid) + 3)]
    initial = [p for p in candidates if edge_free(start, p)]
    if not initial or not free(goal):
        return None
    queue, costs, parent = [], {}, {}
    for p in initial:
        costs[p], parent[p] = math.dist(start, p), None
        heapq.heappush(queue, (costs[p] + math.dist(p, goal), p))
    found = None
    while queue:
        _, p = heapq.heappop(queue)
        if math.dist(p, goal) <= 2 * grid and edge_free(p, goal):
            found = p; break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            q = p[0] + dx * grid, p[1] + dy * grid
            cost = costs[p] + math.hypot(dx, dy) * grid
            if cost < costs.get(q, math.inf) and edge_free(p, q):
                costs[q], parent[q] = cost, p
                heapq.heappush(queue, (cost + math.dist(q, goal), q))
    if found is None:
        return None
    rev = [goal, found]
    while parent[found] is not None:
        found = parent[found]; rev.append(found)
    raw = [start] + list(reversed(rev))
    simplified, index = [raw[0]], 0
    while index < len(raw) - 1:
        end = len(raw) - 1
        while end > index + 1 and not edge_free(raw[index], raw[end]):
            end -= 1
        simplified.append(raw[end]); index = end
    return Route(simplified, smooth=False)


def _run_sidewalk(model, cfg, policy="learned_with_fallback", capture=True):
    cfg.validate()
    world = make_sidewalk_world(cfg.scenario, domain="robot")
    rng, robots, observers = np.random.default_rng(cfg.seed), [], []
    routes = [Route(points, smooth=False) for points in world["routes"]]
    for i in range(cfg.fleet):
        ri = i % len(routes); route = routes[ri]
        x, y, heading = route.pose(0.)
        body = state(x, y, length=.9, width=.65, heading=heading)
        if not footprint_in_walkable(body, world):
            raise ValueError("robot_spawn_footprint_outside_walkway_or_crosswalk")
        robots.append({"id": i, **body, "kind": "delivery_robot", "route": route, "route_index": ri,
            "goal": route.points[-1], "route_progress": 0., "start": (i // len(routes)) * 6.,
            "speed": 1.2 * rng.uniform(.92, 1.08), "active": False, "done": False, "status": "pending",
            "history": deque([dict(body) for _ in range(11)], maxlen=11), "score": None, "eligible": False,
            "blocked_s": 0., "wait_s": 0., "requested": False, "served": False, "request_time": None,
            "route_replans": 0, "replan_failed": False, "known_obstacles": {}, "visible_count": 0,
            "hidden_count": 0, "visible_actor_ids": [], "hidden_actor_ids": [], "unsupported_obstacles": [],
            "domain_blocked": False, "stalled_s": 0., "robot_blockage_s": 0., "max_stalled_s": 0.,
            "request_count": 0, "request_reason": None, "next_request_allowed_t": 0.,
            "planning_failure_count": 0, "consecutive_planning_failures": 0,
            "hold_until": 0., "awaiting_onsite": False, "assisted_travel_m": 0.,
            "assistance_incident_id": None, "assistance_action": None, "assistance_execution_state": None})
        observers.append(ObservationHistory())
    operators = [{"id": i, "vehicle_id": None, "remaining": 0.} for i in range(cfg.operators)]
    fifo, frames, events, score_series = [], [], [], []
    module_active = cfg.assistance_module_enabled and policy == "learned_with_fallback"
    incidents, current_incidents = [], {}
    def incident_state(robot, t, value, message):
        robot["assistance_execution_state"] = value
        if robot["id"] in current_incidents:
            record_incident_state(current_incidents[robot["id"]], t, value, message)
    requests = occupied_s = score_count = out_of_range = 0
    near_frames = contact_frames = static_contacts = hidden_frames = off_walkway = cars_off_road = people_off_walkway = 0
    stationary_contacts = domain_stops = 0
    contact_kinds = {}
    requests_by_reason = {}
    last_fleet_request_t = -math.inf
    queue_admission_deferrals = max_queue_length = 0
    parked = [o for o in world.get("obstacles", []) if o.get("kind") == "parked_car"]
    def event(t, robot, message, **details):
        if capture:
            events.append({"t": round(t, 2), "id": robot["id"], "message": message, **details})
    def observed_actors(t):
        return sidewalk_actors(t, cfg.scenario, seed=cfg.seed, density=cfg.density)
    for tick in range(int(round(cfg.duration_s / .1))):
        t = round(tick * .1, 8)
        actors = observed_actors(t)
        for r in robots:
            if not r["active"] and t >= r["start"] and all(not o["active"] or o["done"] or o["id"] == r["id"] or _gap(r, o, .8) >= .8 for o in robots):
                r["active"] = True; r["status"] = "moving"
        traffic = [{**_body(o), "id": o["id"], "type": 1, "kind": "delivery_robot"}
                   for o in robots if o["active"] and not o["done"]] + actors + [{**o, "type": 1} for o in parked]
        visible = {}
        for r, obs in zip(robots, observers):
            if not r["active"] or r["done"]:
                continue
            obs.update(r, traffic, world.get("buildings", []), cfg.occlusion)
            r["visible_actor_ids"], r["hidden_actor_ids"] = list(obs.visible_ids), list(obs.hidden_ids)
            r["visible_count"], r["hidden_count"] = len(obs.visible_ids), len(obs.hidden_ids)
            visible[r["id"]] = [o for o in traffic if o["id"] in obs.visible_ids]
            for obj in world.get("obstacles", []):
                if math.hypot(obj["x"] - r["x"], obj["y"] - r["y"]) <= 35 and line_of_sight(r, obj, world.get("buildings", []), cfg.occlusion):
                    r["known_obstacles"][obj["id"]] = obj
            r["unsupported_obstacles"] = [ident for ident, o in r["known_obstacles"].items() if o.get("kind") != "parked_car"]
            if not obs.visible_ids or any(_gap(r, o, 2.) <= 2. for o in visible[r["id"]]):
                r["score"] = None; r["eligible"] = False
        if tick % 10 == 0 and (policy in ("learned", "learned_with_fallback", "fallback_only") or capture):
            targets, rows = [], []
            for r, obs in zip(robots, observers):
                if not r["active"] or r["done"]:
                    continue
                try:
                    rows.append(model.features_from_scene(obs.scene(r, world, t))); targets.append(r)
                except ValueError as exc:
                    if str(exc) != "no_current_neighbors":
                        raise
                    r["score"] = None; r["eligible"] = False
            outputs = model.predict_features_many(rows) if rows else []
            for r, pred in zip(targets, outputs):
                r["eligible"] = bool(pred["score_applicable"])
                r["score"] = float(pred["score"]) if r["eligible"] else None
                score_count += 1; out_of_range += bool(pred["outside_training_range_features"])
        for operator in operators:
            if operator["vehicle_id"] is None:
                continue
            occupied_s += .1; operator["remaining"] = max(0., operator["remaining"] - .1)
            if operator["remaining"] <= 1e-8:
                r = robots[operator["vehicle_id"]]; r["served"] = True
                if module_active:
                    incident = current_incidents[r["id"]]
                    decision = plan_assistance(r, world, list(r["known_obstacles"].values()),
                        visible.get(r["id"], []), plan_sidewalk_detour, cfg.assistance_action)
                    incident.update(finished_at=round(t, 3), action=decision.action,
                        instructions=list(decision.instructions), explanation=decision.explanation,
                        plan_failure=decision.plan_failure, decision_source="scenario_rules" if cfg.assistance_action=="auto" else "user_selected_simulation_command",
                        visible_actor_ids_at_review=list(r["visible_actor_ids"]),
                        known_static_ids_at_review=list(r["known_obstacles"]))
                    r["assistance_action"] = decision.action
                    if decision.route is not None:
                        r["route"], r["route_progress"] = decision.route, 0.; r["route_replans"] += 1
                        r["consecutive_planning_failures"] = 0
                        incident["route_points"] = [[float(x), float(y)] for x, y in decision.route.points]
                        incident_state(r, t, "route_executing", "已下发新路点；机器人按原通行约束执行")
                    elif decision.action == "hold_then_resume":
                        r["hold_until"] = t + decision.hold_s
                        incident["hold_until"] = round(r["hold_until"], 3)
                        incident_state(r, t, "holding", "已下发等待让行指令；3秒后重新检查")
                    else:
                        r["awaiting_onsite"] = True
                        if decision.plan_failure:
                            r["replan_failed"] = True; r["planning_failure_count"] += 1; r["consecutive_planning_failures"] += 1
                        incident_state(r, t, "onsite_required", "保持停车，升级现场处理；本模拟不安排虚构的现场清障")
                    event(t, r, "模拟协助员下发指令："+ACTION_LABELS[decision.action],
                        assistance_action=decision.action, incident_id=incident["id"])
                else:
                    route = plan_sidewalk_detour(r, world, list(r["known_obstacles"].values()))
                    if route is None:
                        r["replan_failed"] = True
                        r["planning_failure_count"] += 1
                        r["consecutive_planning_failures"] += 1
                        event(t, r, "保守规划器未找到可接受的步道绕路；保持通行约束")
                    else:
                        r["route"], r["route_progress"] = route, 0.; r["route_replans"] += 1
                        r["consecutive_planning_failures"] = 0
                        event(t, r, "协助完成，尝试规则 A* 步道/斑马线绕行路线")
                r["next_request_allowed_t"] = t + FALLBACK_RULES["retry_cooldown_after_service_s"]
                operator["vehicle_id"] = None
        proposals = []
        for r in robots:
            if not r["active"] or r["done"]:
                continue
            busy = any(o["vehicle_id"] == r["id"] for o in operators)
            progress = min(r["route"].total, r["route_progress"] + r["speed"] * .1)
            x, y, heading = r["route"].pose(progress)
            guards = [{**o, "x": o["x"] + o.get("vx", 0.) * .1, "y": o["y"] + o.get("vy", 0.) * .1}
                      for o in visible[r["id"]]]
            static = world.get("buildings", []) + list(r["known_obstacles"].values())
            safe, in_domain = True, True
            for u in (.25, .5, .75, 1.):
                xx, yy, hh = r["route"].pose(r["route_progress"] + (progress - r["route_progress"]) * u)
                angle_delta = (hh - r["heading"] + math.pi) % (2 * math.pi) - math.pi
                body = state(xx, yy, length=.9, width=.65, heading=r["heading"] + u * angle_delta)
                if not footprint_in_walkable(body, world):
                    in_domain = False; safe = False; break
                if any(_gap(body, o, .4) < .4 for o in guards) or any(_gap(body, o, .2) < .2 for o in static):
                    safe = False; break
            r["domain_blocked"] = not in_domain
            domain_stops += not in_domain
            commanded_hold = t < r["hold_until"] - 1e-8
            blocked = not safe or busy or r["id"] in fifo or commanded_hold or r["awaiting_onsite"]
            pending = busy or r["id"] in fifo or commanded_hold or r["awaiting_onsite"]
            if pending:
                r["stalled_s"] = r["robot_blockage_s"] = 0.
            elif blocked:
                r["stalled_s"] += .1
                r["max_stalled_s"] = max(r["max_stalled_s"], r["stalled_s"])
                trial = state(x, y, length=.9, width=.65, heading=heading)
                robot_blocked = math.hypot(r["vx"], r["vy"]) < .05 and any(
                    o.get("kind") == "delivery_robot" and math.hypot(o.get("vx", 0.), o.get("vy", 0.)) < .05
                    and _gap(trial, o, .4) < .4 for o in guards)
                r["robot_blockage_s"] = r["robot_blockage_s"] + .1 if robot_blocked else 0.
            else:
                r["stalled_s"] = r["robot_blockage_s"] = 0.
            if blocked:
                r["blocked_s"] += .1; r["wait_s"] += .1
                proposals.append((r, r["x"], r["y"], r["heading"], r["route_progress"], True, busy))
            else:
                r["blocked_s"] = 0.
                proposals.append((r, x, y, heading, progress, False, busy))
        for r, x, y, heading, progress, blocked, busy in proposals:
            travelled = math.hypot(x-r["x"], y-r["y"])
            if r["assistance_action"]:
                r["assisted_travel_m"] += travelled
            r["vx"], r["vy"] = (x - r["x"]) / .1, (y - r["y"]) / .1
            r["x"], r["y"], r["heading"], r["route_progress"] = x, y, heading, progress
            if progress >= r["route"].total - 1e-6:
                r["done"] = True; r["status"] = "done"; r["score"] = None; r["eligible"] = False
                r["vx"] = r["vy"] = 0.; event(t + .1, r, "配送机器人到达人行道模拟终点")
                if r["id"] in current_incidents:
                    incident_state(r, t + .1, "completed", "机器人实际到达任务终点")
            else:
                r["status"] = "onsite_required" if r["awaiting_onsite"] else "holding" if t < r["hold_until"]-1e-8 else "assisting" if busy else "queued" if r["id"] in fifo else "waiting" if blocked else "bypassing" if r["route_replans"] else "moving"
                if r["id"] in current_incidents and not busy and r["id"] not in fifo and not r["awaiting_onsite"] and t >= r["hold_until"]-1e-8:
                    if travelled > 1e-8:
                        next_state = "route_executing" if r["assistance_action"] == "replan_walkway" else "resumed"
                        incident_state(r, t, next_state, "机器人按指令实际前进，通行检查保持启用")
                    elif r["stalled_s"] >= 2.-1e-8:
                        incident_state(r, t, "blocked_after_instruction", "收到指令后仍受阻，继续检测并保留未解决状态")
            pending = r["id"] in fifo or any(o["vehicle_id"] == r["id"] for o in operators) or t < r["hold_until"]-1e-8 or r["awaiting_onsite"]
            reason = assistance_request_reason(r, policy, cfg, t, pending)
            if reason:
                hybrid = uses_stall_fallback(policy, cfg)
                capacity = FALLBACK_RULES["waiting_queue_slots_per_operator"] * cfg.operators
                if hybrid and (len(fifo) >= capacity or t - last_fleet_request_t < FALLBACK_RULES["minimum_fleet_request_interval_s"] - 1e-8):
                    queue_admission_deferrals += 1
                    continue
                requests += 1; r["requested"] = True; r["request_time"] = t; fifo.append(r["id"])
                r["request_count"] += 1; r["request_reason"] = reason
                if module_active:
                    previous = current_incidents.get(r["id"])
                    if previous:
                        record_incident_state(previous, t, "replaced_by_retry", "再次请求协助；前一条指令未结束任务")
                    incident = new_incident(r, reason, t)
                    incident.update(blocked_s_at_request=round(r["stalled_s"], 3), robot_blockage_s_at_request=round(r["robot_blockage_s"], 3),
                        planning_failures_at_request=r["consecutive_planning_failures"])
                    incidents.append(incident); current_incidents[r["id"]] = incident
                    r["assistance_incident_id"] = incident["id"]
                    r["assistance_action"] = None
                    r["assistance_execution_state"] = "queued"
                r["next_request_allowed_t"] = t + FALLBACK_RULES["retry_cooldown_after_service_s"]
                requests_by_reason[reason] = requests_by_reason.get(reason, 0) + 1
                last_fleet_request_t = t; max_queue_length = max(max_queue_length, len(fifo))
                messages = {"model_threshold": "学习模型分数门控请求模拟协助", "legacy_wait_rule": "等待规则请求模拟协助",
                    "persistent_stall": "独立规则：持续停滞请求协助", "robot_blockage": "独立规则：机器人相互堵塞请求协助",
                    "repeated_planning_failure": "独立规则：重复规划失败请求再次协助"}
                event(t, r, messages[reason], request_reason=reason, request_count=r["request_count"],
                    blocked_s=round(r["stalled_s"], 2), robot_blockage_s=round(r["robot_blockage_s"], 2),
                    score=r["score"], eligible=r["eligible"], planning_failure_count=r["planning_failure_count"])
                r["stalled_s"] = r["robot_blockage_s"] = 0.
        for r in robots:
            r["history"].append(_body(r))
        for operator in operators:
            if operator["vehicle_id"] is None and fifo:
                ident = fifo.pop(0); operator.update(vehicle_id=ident, remaining=cfg.service_s)
                robots[ident]["status"] = "assisting"; event(t, robots[ident], "模拟操作员开始处理")
                if module_active:
                    current_incidents[ident]["started_at"] = round(t, 3)
                    incident_state(robots[ident], t, "reviewing", "模拟协助员占用席位，检查当前观测与已知地图")
        if (tick + 1) % 5:
            continue
        truth_actors = observed_actors(round(t + .1, 8))
        active = [r for r in robots if r["active"] and not r["done"]]
        off_walkway += sum(not footprint_in_walkable(r, world) for r in robots if r["active"])
        cars_off_road += sum(not footprint_in_road(a, world) for a in truth_actors if a.get("type") == 1)
        cars_off_road += sum(not footprint_in_road(a, world) for a in parked)
        people_off_walkway += sum(not footprint_in_walkable(a, world) for a in truth_actors if a.get("type") == 2)
        contact_ids, contact_pairs = [], []
        for r in active:
            truth_static = world.get("buildings", []) + world.get("obstacles", [])
            truth = [{**o, "kind": "delivery_robot"} for o in active if o["id"] != r["id"]] + truth_actors + truth_static
            contacts = [o for o in truth if _gap(r, o, .000001) <= 1e-8]
            near_frames += any(_gap(r, o, .1) <= .1 for o in truth)
            contact_frames += bool(contacts); hidden_frames += r["hidden_count"]
            static_contacts += any(_gap(r, o, .000001) <= 1e-8 for o in truth_static)
            if contacts:
                contact_ids.append(r["id"]); stationary_contacts += math.hypot(r["vx"], r["vy"]) < .05
                for kind in set(o.get("kind") or {1:"car", 2:"pedestrian"}.get(o.get("type"), "unknown") for o in contacts):
                    contact_kinds[kind] = contact_kinds.get(kind, 0) + 1
                contact_pairs += [{"vehicle_id": r["id"], "object_id": o["id"], "kind": o.get("kind", "unknown"),
                                   "vehicle_stationary": math.hypot(r["vx"], r["vy"]) < .05} for o in contacts]
        if capture:
            main = robots[0]
            frames.append({"t": round(t + .1, 2), "vehicles": [{**_public_vehicle(r), "kind": "delivery_robot", "domain_blocked": r["domain_blocked"],
                    "request_reason": r["request_reason"], "request_count": r["request_count"], "blocked_s": round(r["stalled_s"], 2),
                    "robot_blockage_s": round(r["robot_blockage_s"], 2), "planning_failure_count": r["planning_failure_count"],
                    "consecutive_planning_failures": r["consecutive_planning_failures"],
                    "assistance_incident_id": r["assistance_incident_id"], "assistance_action": r["assistance_action"],
                    "assistance_execution_state": r["assistance_execution_state"], "assisted_travel_m": round(r["assisted_travel_m"], 4)} for r in robots if r["active"]],
                "actors": truth_actors, "queue": [{"id": i, "score": robots[i]["score"]} for i in fifo],
                "operators": [dict(o) for o in operators], "completed": sum(r["done"] for r in robots),
                "visible_actor_ids": list(main["visible_actor_ids"]), "hidden_actor_ids": list(main["hidden_actor_ids"]),
                "contact_vehicle_ids": contact_ids, "contact_pairs": contact_pairs})
            score_series.append({"t": round(t + .1, 2), "score": main["score"], "threshold": cfg.threshold, "eligible": main["eligible"]})
    waits = [r["wait_s"] for r in robots]
    summary = {"completed": sum(r["done"] for r in robots), "unfinished": sum(not r["done"] for r in robots), "total": cfg.fleet,
        "mean_wait_s": round(float(np.mean(waits)), 2), "p95_wait_s": round(float(np.percentile(waits, 95)), 2),
        "requests": int(requests), "operator_busy_fraction": min(1., occupied_s / (cfg.operators * cfg.duration_s)),
        "requests_by_reason": requests_by_reason,
        "model_requests": requests_by_reason.get("model_threshold", 0),
        "fallback_requests": sum(requests_by_reason.get(k, 0) for k in ("persistent_stall", "robot_blockage", "repeated_planning_failure")),
        "retry_requests": sum(max(0, r["request_count"] - 1) for r in robots),
        "max_requests_per_robot": max(r["request_count"] for r in robots),
        "max_continuous_stall_s": round(max(r["max_stalled_s"] for r in robots), 2),
        "planning_failure_attempts": sum(r["planning_failure_count"] for r in robots),
        "max_queue_length": max_queue_length, "queue_admission_deferrals": queue_admission_deferrals,
        "assistance_actions": {action: sum(i["action"] == action for i in incidents) for action in ACTION_LABELS},
        "onsite_required_robots": sum(r["awaiting_onsite"] for r in robots),
        "assisted_robots_arrived": sum(r["done"] and r["request_count"]>0 for r in robots),
        "travel_after_assistance_m": round(sum(r["assisted_travel_m"] for r in robots), 3),
        "near_contact_frames": int(near_frames), "physical_overlap_frames": int(contact_frames), "static_contact_frames": int(static_contacts),
        "contact_vehicle_frames_by_object_kind": contact_kinds, "contact_stationary_vehicle_frames": int(stationary_contacts),
        "off_walkway_frames": int(off_walkway), "cars_off_road_frames": int(cars_off_road), "pedestrians_off_walkway_frames": int(people_off_walkway),
        "domain_guard_stop_steps": int(domain_stops), "occluded_actor_frames": int(hidden_frames),
        "unsupported_static_objects": len(world.get("buildings", [])) + sum(o.get("kind") != "parked_car" for o in world.get("obstacles", [])),
        "waiting_at_end": sum(r["active"] and not r["done"] and r["status"] in ("waiting", "assisting", "queued", "holding", "onsite_required") for r in robots),
        "route_replans": sum(r["route_replans"] for r in robots), "failed_replans": sum(r["replan_failed"] for r in robots),
        "fraction_scored_frames_outside_training_minmax": out_of_range / score_count if score_count else None,
        "domain_warning": "人行道配送机器人尺寸、速度、步道地图均属原道路车训练域外；原模型分数未经校准，空间域与避让为独立规则", "policy": policy}
    return {"config": asdict(cfg), "world": world, "frames": frames, "events": events, "score_series": score_series, "summary": summary,
        "assistance_incidents": incidents,
        "metadata": {"synthetic": True, "world_kind": "sidewalk_delivery_robots_with_separate_motor_road",
            "model_scores": "actual_frozen_hist_gradient_boosting_predict_proba", "score_role": "request_gate_only; not learned motion or danger probability",
            "robot_size_m": [.9, .65], "robot_nominal_speed_mps": 1.2, "domain_ood": True,
            "robot_neighbors_encoded_as_vehicle_kinematic_proxy": True,
            "observation": "actual 0.1s per-ego history; current building LOS only; no hidden-history backfill",
            "routing": "supplied sidewalk/crosswalk route; assistance uses conservative static A* exclusively inside walkway/crosswalk union",
            "forbidden_route_domain": "motor roads outside crosswalk polygons; robots never use roads as a detour",
            "safety_guard": "independent rule: full footprint inside walkable union; .4m visible dynamic clearance, .2m mapped/discovered static clearance",
            "static_map_assumption": "buildings and walkable domain known; other obstacles discovered through LOS within35m",
            "overlap_metric": "0.5s robot frames with true rectangle contact/overlap, not unique crashes",
            "domain_metric": "0.5s full-footprint frames outside permitted walkable union/road union",
            "one_request_per_vehicle": not uses_stall_fallback(policy, cfg), "queue": "FIFO", "service_duration_s": cfg.service_s,
            "fallback_enabled": uses_stall_fallback(policy, cfg),
            "fallback_rules_are_engineering_assumptions": True, "fallback_rules": dict(FALLBACK_RULES),
            "fallback_policy": dict(FALLBACK_RULES),
            "assistance_policy": {"module_version": ASSISTANCE_VERSION, "enabled": module_active,
                "decision_source": "scenario_rules_or_user_selected_simulation_command",
                "actions": dict(ACTION_LABELS), "hold_s": 3., "review_radius_m": 4.,
                "onsite_arrival_simulated": False, "motion_guard_override": False,
                "planning_failure_does_not_prove_real_world_has_no_route": True},
            "blocked_timer_semantics": "non-progress outside queue/service/commanded-hold/onsite-escalation only; progress resets timer",
            "not_validated": ["real safety", "traffic priority", "assistance need", "sensor realism", "delivery economics", "probability calibration"]}}


def simulate_sidewalk(model, config=None, compare=True):
    cfg = config if isinstance(config, SidewalkConfig) else SidewalkConfig(**(config or {}))
    policy = "learned_with_fallback" if cfg.fallback_enabled else "learned"
    result = _run_sidewalk(model, cfg, policy, True)
    if compare:
        names = {"learned_with_fallback": "模型 + 停滞兜底 + 协助指令", "learned": "原纯模型门控（无兜底）", "fallback_only": "仅加停滞求助（旧处理器）", "all": "等待2秒即请求", "no_assist": "仅自主等待"}
        comparisons = ("learned", "fallback_only", "all", "no_assist") if cfg.fallback_enabled else ("all", "no_assist")
        summaries = [result["summary"]] + [_run_sidewalk(model, cfg, p, False)["summary"] for p in comparisons]
        result["policies"] = [{"name": names[row["policy"]], **row} for row in summaries]
    return result
