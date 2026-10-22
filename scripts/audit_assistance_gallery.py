"""Audit saved assistance lifecycles and actual motion without controller imports."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, default=root/"demo/sidewalk_gallery.js")
    parser.add_argument("--output", type=Path, default=root/"results/sidewalk/assistance_audit.json")
    args = parser.parse_args()
    raw = args.gallery.read_bytes()
    gallery = json.loads(raw.decode("utf-8").split("=",1)[1].strip().removesuffix(";"))
    rows, errors = [], []
    for name, run in gallery["runs"].items():
        def require(condition, message):
            if not condition:
                errors.append(f"{name}: {message}")
        cfg, summary = run["config"], run["summary"]
        rules, incidents = run["metadata"]["fallback_policy"], run["assistance_incidents"]
        require(run["metadata"]["assistance_policy"]["enabled"] is True, "operator module must be active")
        require(len(incidents) == summary["requests"], "request count must equal incident count")
        require(len({i["id"] for i in incidents}) == len(incidents), "duplicate incident id")
        reasons = Counter(i["request_reason"] for i in incidents)
        require(dict(reasons) == summary["requests_by_reason"], "reason totals differ")
        request_events = [e for e in run["events"] if "request_reason" in e]
        require(len(request_events) == len(incidents), "request event count differs")
        require(Counter((e["id"], e["request_reason"], round(e["t"],2)) for e in request_events) ==
                Counter((i["robot_id"], i["request_reason"], round(i["requested_at"],2)) for i in incidents),
                "request events do not join their incidents")
        per_robot, review_edges = defaultdict(list), []
        for incident in incidents:
            per_robot[incident["robot_id"]].append(incident)
            reason = incident["request_reason"]
            if reason == "persistent_stall":
                require(incident["blocked_s_at_request"] >= rules["persistent_stall_s"]-1e-6, "stall requested too early")
            elif reason == "robot_blockage":
                require(incident["robot_blockage_s_at_request"] >= rules["stationary_robot_blockage_s"]-1e-6, "robot block requested too early")
            elif reason == "repeated_planning_failure":
                require(incident["planning_failures_at_request"] >= rules["consecutive_planning_failures"], "failure requested too early")
            elif reason == "model_threshold":
                require(incident["eligible_at_request"] and incident["score_at_request"] is not None and incident["score_at_request"] >= cfg["threshold"], "model request has no valid high score")
            else:
                require(False, "unknown primary request reason")
            if reason in ("persistent_stall", "robot_blockage"):
                seconds = rules["persistent_stall_s"] if reason=="persistent_stall" else rules["stationary_robot_blockage_s"]
                window = [b for f in run["frames"] for b in f["vehicles"] if b["id"]==incident["robot_id"]
                          and incident["requested_at"]-seconds+.500001 <= f["t"] <= incident["requested_at"]+1e-8]
                require(len(window)>=2 and all(math.hypot(b["x"]-window[0]["x"],b["y"]-window[0]["y"])<1e-6 for b in window),
                        "saved coordinates do not support the reported stall")
            events = incident["execution_events"]
            require(bool(events) and events[0]["state"] == "queued", "incident must start queued")
            require(all(a["t"] <= b["t"]+1e-8 for a,b in zip(events,events[1:])), "incident states out of order")
            require(incident["execution_state"] == events[-1]["state"], "final state differs from last event")
            start, finish = incident["started_at"], incident["finished_at"]
            if start is not None:
                require(start >= incident["requested_at"]-1e-8, "review starts before request")
                review_edges.append((start,1))
                review_edges.append((finish if finish is not None else cfg["duration_s"]+.1,-1))
            if finish is not None:
                require(start is not None and abs(finish-start-cfg["service_s"]) < 1e-6, "review duration differs from configured service")
                require(bool(incident["instructions"]) and incident["action"] in ("replan_walkway","hold_then_resume","onsite_required"), "finished review needs real command")
                if incident["action"] == "replan_walkway":
                    require(len(incident["route_points"]) >= 2, "route command has no route")
                if incident["action"] == "hold_then_resume":
                    require(abs(incident["hold_until"]-finish-3.) < 1e-6, "yield command must hold three seconds")
        concurrent = 0
        for t,delta in sorted(review_edges):
            concurrent += delta
            require(0 <= concurrent <= cfg["operators"], "operator capacity exceeded")
        request_times = sorted(i["requested_at"] for i in incidents)
        require(all(b-a >= rules["minimum_fleet_request_interval_s"]-1e-6 for a,b in zip(request_times,request_times[1:])), "fleet request interval too short")
        for robot_incidents in per_robot.values():
            require(len(robot_incidents) <= rules["max_requests_per_robot"], "request cap exceeded")
            for a,b in zip(robot_incidents,robot_incidents[1:]):
                require(a["finished_at"] is not None and b["requested_at"]-a["finished_at"] >= rules["retry_cooldown_after_service_s"]-1e-6, "retry before post-service cooldown")
                require(a["action"] != "onsite_required", "new request after onsite escalation")
        previous, moving_frames = {}, 0
        for frame in run["frames"]:
            queue = [q["id"] for q in frame["queue"]]
            serving = [o["vehicle_id"] for o in frame["operators"] if o["vehicle_id"] is not None]
            require(len(queue) == len(set(queue)) and len(serving) == len(set(serving)) and not set(queue)&set(serving), "queue/service duplicates")
            require(len(queue) <= rules["waiting_queue_slots_per_operator"]*cfg["operators"], "waiting capacity exceeded")
            bodies = {b["id"]:b for b in frame["vehicles"]}
            for ident, body in bodies.items():
                if ident in previous:
                    old_t, old = previous[ident]
                    distance = math.hypot(body["x"]-old["x"],body["y"]-old["y"])
                    require(distance <= 1.2*1.08*(frame["t"]-old_t)+1e-6, "position jump exceeds physical speed limit")
                    require(body["assisted_travel_m"]+1e-6 >= old["assisted_travel_m"], "assisted travel counter decreased")
                    if body["status"] in ("holding","onsite_required") and old["status"]==body["status"] and old["assistance_incident_id"]==body["assistance_incident_id"]:
                        require(distance < 1e-6, "a held robot changed saved coordinates")
                    moving_frames += body["assistance_action"] is not None and distance > 1e-6
                if ident in queue or ident in serving or body["status"] in ("holding","onsite_required"):
                    require(body["blocked_s"] == 0., "commanded/queued hold counted as a new stall")
                if body["status"] in ("holding","onsite_required"):
                    require(math.hypot(body["vx"],body["vy"]) < 1e-8 and not body["done"], "hold/onsite fakes motion or completion")
                previous[ident] = frame["t"],body
            for incident in incidents:
                finish = incident["finished_at"]
                body = bodies.get(incident["robot_id"])
                if finish is None or body is None or frame["t"] < finish+.100001:
                    continue
                if incident["action"] == "onsite_required":
                    require(body["status"] == "onsite_required" and not body["done"], "onsite command must remain unresolved")
                if incident["action"] == "hold_then_resume" and frame["t"] < incident["hold_until"]-1e-6:
                    require(body["status"] == "holding", "yield command not physically held")
        actions = Counter(i["action"] for i in incidents if i["action"] is not None)
        require(all(summary["assistance_actions"].get(k)==v for k,v in actions.items()), "command totals differ")
        rows.append({"scenario":name, "requests":len(incidents), "request_reasons":dict(reasons),
            "unscorable_fallback_requests":sum(i["request_reason"]!="model_threshold" and not i["eligible_at_request"] for i in incidents),
            "commands_issued":sum(i["finished_at"] is not None for i in incidents), "actions":dict(actions),
            "post_command_moving_robot_frames":moving_frames,
            "assisted_robots_arrived":summary["assisted_robots_arrived"],
            "onsite_required_robots":summary["onsite_required_robots"],
            "travel_after_assistance_m":summary["travel_after_assistance_m"]})
    result = {"status":"pass" if not errors else "fail", "gallery_sha256":hashlib.sha256(raw).hexdigest(),
        "independent_implementation":"saved incident timestamps, queues and actual coordinates; no controller imports",
        "scene_count":len(rows), "scenes":rows, "errors":errors,
        "not_a_real_human_operator_or_safety_evaluation":True}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"scenes":len(rows),"requests":sum(r["requests"] for r in rows),"errors":errors[:10]},ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
