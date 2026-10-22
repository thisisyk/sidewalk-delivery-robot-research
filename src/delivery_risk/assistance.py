"""Explicit simulated operator decisions; no real remote-control connection.

Only the current visible actors and mapped/discovered static context are used.
The module proposes a command. The simulator still checks every motion step;
operator review neither removes obstacles nor overrides the walking domain.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable


ASSISTANCE_VERSION = "sidewalk_operator_v2_visible_robot_blockers"
ASSISTANCE_ACTIONS = ("auto", "replan_walkway", "hold_then_resume", "onsite_required")
ACTION_LABELS = {
    "replan_walkway": "批准步道内重新规划",
    "hold_then_resume": "暂停让行后重新检查",
    "onsite_required": "保持停车并升级现场处理",
}


@dataclass(frozen=True)
class AssistanceDecision:
    action: str
    explanation: str
    instructions: tuple[str, ...]
    route: object | None = None
    hold_s: float = 0.
    plan_failure: str | None = None


def plan_assistance(robot: dict, world: dict, known_obstacles: list[dict],
                    visible_actors: list[dict], planner: Callable,
                    action: str = "auto") -> AssistanceDecision:
    """Choose a reproducible simulated review action, never a risk threshold.

    A user-selected action may override the scenario rule, but cannot authorize
    crossing the motor road or disabling collision checks.  The three-second
    hold and four-metre review radius are engineering assumptions, not learned
    values or validated traffic rules.
    """
    if action not in ASSISTANCE_ACTIONS:
        raise ValueError("unsupported_assistance_action")
    if action == "auto":
        nearby = [a for a in visible_actors
                  if math.hypot(a["x"]-robot["x"], a["y"]-robot["y"]) <= 4.]
        moving_pedestrian = any(a.get("type") == 2 and math.hypot(a.get("vx", 0.), a.get("vy", 0.)) > .05 for a in nearby)
        robot_blocker = any(a.get("kind") == "delivery_robot" for a in nearby)
        action = "hold_then_resume" if moving_pedestrian and not robot_blocker else "replan_walkway"
    if action == "hold_then_resume":
        return AssistanceDecision(action,
            "当前可见的移动行人可能短时占用通道；先让行，再恢复原路线并重新检查通行条件。",
            ("保持当前位置，不关闭避障检查", "等待3秒让行", "重新检查当前观测，允许通行才沿原路线继续"), hold_s=3.)
    if action == "onsite_required":
        return AssistanceDecision(action,
            "当前远程方案不继续尝试通行；请求现场检查或清障，本模拟不伪造现场人员到达。",
            ("保持停车", "将任务标记为需要现场处理", "等待现场检查，不移动到终点"))
    # A stopped robot is the actual obstruction in a mutual blockage.  The
    # operator must plan around its CURRENT visible footprint, not approve the
    # same static-map path through it. Moving/hidden robots are not frozen into
    # this snapshot; all actors remain subject to the normal per-step guard.
    stationary_robots = [a for a in visible_actors if a.get("kind") == "delivery_robot"
        and a.get("id") != robot["id"] and math.hypot(a.get("vx", 0.), a.get("vy", 0.)) < .05]
    route = planner(robot, world, list(known_obstacles)+stationary_robots)
    if route is None:
        return AssistanceDecision("onsite_required",
            "保守规划器未找到可接受的步道路线；这可能是规划器局限，不证明现实环境无路。",
            ("保持停车", "记录本次规划未成功", "升级现场检查或清障；任务保留未完成"),
            plan_failure="conservative_planner_returned_no_route")
    return AssistanceDecision("replan_walkway",
        f"根据已知建筑、已发现障碍及{len(stationary_robots)}台当前可见静止机器人的占地生成步道新路线。动态对象仍逐步触发避让。",
        ("下发新的步道/斑马线路点", "按原速度上限与逐步通行检查执行", "若再次受阻，记录停滞并按冷却和次数限制重新请求"), route=route)


def new_incident(robot: dict, reason: str, t: float) -> dict:
    incident = {
        "id": f"robot_{robot['id']}_request_{robot['request_count']}",
        "robot_id": robot["id"], "request_reason": reason,
        "requested_at": round(t, 3), "started_at": None, "finished_at": None,
        "score_at_request": robot.get("score"), "eligible_at_request": bool(robot.get("eligible")),
        "action": None, "instructions": [], "route_points": [], "hold_until": None,
        "plan_failure": None, "decision_source": None,
        "observation_source": "current visible actors and mapped/discovered static objects only",
        "execution_state": "queued", "execution_events": [],
    }
    record_incident_state(incident, t, "queued", "协助请求进入FIFO队列")
    return incident


def record_incident_state(incident: dict, t: float, state: str, message: str) -> None:
    events = incident["execution_events"]
    if events and events[-1]["state"] == state:
        return
    if events and t < events[-1]["t"] - 1e-7:
        raise ValueError("assistance_incident_time_must_be_monotonic")
    incident["execution_state"] = state
    events.append({"t": round(t, 3), "state": state, "message": message})
