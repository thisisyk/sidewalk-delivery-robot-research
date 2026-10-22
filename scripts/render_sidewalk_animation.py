"""Render a four-panel sidewalk-robot GIF from saved model-scored replays.

This script never invokes the model or reconstructs actors. It uses only saved
frame coordinates, headings, eligibility, scores and contact/visibility fields.
The JavaScript input must be a JSON assignment, not executable JavaScript.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


SCENES = ("crowd", "blind_corner", "trash", "mixed")
TITLES = {"crowd": "移动人群 · 停滞求助", "blind_corner": "盲角 · 人行道转弯", "trash": "人行道 · 垃圾堆占道", "mixed": "综合街角 · 步道多障碍"}
COLORS = {"paper": "#edf3ed", "ink": "#19383d", "muted": "#647d76", "map": "#132d36", "road": "#2c464b", "green": "#94d5b2", "wait": "#ffc57a", "assist": "#83c6e6", "contact": "#ff8b76"}
WIDTH, HEIGHT = 1120, 1060


ACTION_LABELS = {"replan_walkway": "步道内重新规划", "hold_then_resume": "停候后恢复", "onsite_required": "保持停车，待现场处理"}
STATE_LABELS = {"queued": "已请求，队列等待", "reviewing": "模拟协助员复核中", "route_executing": "新路线下发，尝试执行", "holding": "协助指令：等待让行", "resumed": "恢复指令后自主运行", "completed": "已到模拟终点", "blocked_after_instruction": "指令后再次受阻", "onsite_required": "停车，需要现场处理", "replaced_by_retry": "前次指令被再次求助替代"}
REASON_LABELS = {"model_threshold": "模型门槛", "persistent_stall": "持续停滞兜底", "robot_blockage": "机器人互堵兜底", "repeated_planning_failure": "连续规划失败兜底", "legacy_wait_rule": "固定等待规则"}


def assistance_at(run, frame):
    """Read only a request/command/state that has happened by this saved frame.

    incident.execution_state and any final travel total are intentionally never
    read: neither would describe the robot at an earlier replay time.
    """
    t = float(frame["t"])
    available = [i for i in run.get("assistance_incidents", [])
                 if isinstance(i.get("requested_at"), (int, float)) and i["requested_at"] <= t+1e-8]
    if not available:
        return None
    # Prefer the most recently issued command so a newer queued request does
    # not conceal its actual subsequent operation. This does not select on
    # success: failed, blocked, holding and onsite commands are all eligible.
    issued_so_far = [i for i in available if isinstance(i.get("finished_at"), (int, float))
                     and i["finished_at"] <= t+1e-8]
    incident = (max(issued_so_far, key=lambda i: (i["finished_at"], i["requested_at"], str(i.get("id", ""))))
                if issued_so_far else max(available, key=lambda i: (i["requested_at"], str(i.get("id", "")))))
    started = isinstance(incident.get("started_at"), (int, float)) and incident["started_at"] <= t+1e-8
    issued = isinstance(incident.get("finished_at"), (int, float)) and incident["finished_at"] <= t+1e-8
    events = sorted([e for e in incident.get("execution_events", [])
                     if isinstance(e.get("t"), (int, float)) and e["t"] <= t+1e-8], key=lambda e: e["t"])
    robot = next((r for r in frame.get("vehicles", []) if str(r.get("id")) == str(incident["robot_id"])), None)
    frame_state = robot.get("assistance_execution_state") if robot and str(robot.get("assistance_incident_id")) == str(incident.get("id")) else None
    state = events[-1].get("state") if events else frame_state or ("unknown" if issued else "reviewing" if started else "queued")
    return {"incident": incident, "issued": issued, "started": started, "state": state,
            "robot": robot, "travel_m": robot.get("assisted_travel_m") if robot else None,
            "event": events[-1] if events else None}


def robot_color(robot):
    status = str(robot.get("status", ""))
    if robot.get("done") or status in ("done", "completed"):
        return "#6b9283"
    if status == "onsite_required":
        return "#be7165"
    if status == "holding":
        return "#c99e48"
    if status in ("waiting", "queued", "blocked"):
        return "#ce9c45"
    if "assist" in status or status == "servicing":
        return "#4f98b9"
    return "#4e9e72"


def command_timeline(run):
    horizon = float(run["frames"][-1]["t"])
    records = []
    for incident in run.get("assistance_incidents", []):
        issued = isinstance(incident.get("finished_at"), (int, float)) and incident["finished_at"] <= horizon
        records.append({"id": incident.get("id"), "robot_id": incident.get("robot_id"),
                        "request_reason": incident.get("request_reason"),
                        "requested_at": incident.get("requested_at"), "started_at": incident.get("started_at"),
                        "issued_at": incident.get("finished_at") if issued else None,
                        "action": incident.get("action") if issued else None,
                        "instructions": incident.get("instructions", []) if issued else [],
                        "plan_failure": incident.get("plan_failure") if issued else None,
                        "execution_events": sorted([dict(e) for e in incident.get("execution_events", [])
                                                    if isinstance(e.get("t"), (int, float)) and e["t"] <= horizon], key=lambda e: e["t"])})
    return records


def read_gallery(path):
    text = path.read_text(encoding="utf-8-sig")
    prefix = re.match(r"\s*(?:window\.)?DELIVERY_SIDEWALK_GALLERY\s*=\s*", text)
    if not prefix:
        raise ValueError("Expected a JSON-only DELIVERY_SIDEWALK_GALLERY assignment")
    value, end = json.JSONDecoder().raw_decode(text, prefix.end())
    if text[end:].strip() not in ("", ";"):
        raise ValueError("Executable content after gallery JSON is not accepted")
    for scenario in SCENES:
        run = value.get("runs", {}).get(scenario)
        if not run or not run.get("frames") or not run.get("world"):
            raise ValueError(f"Missing saved scenario replay: {scenario}")
        if run.get("config", {}).get("domain") != "robot" or run["world"].get("schema") != "sidewalk_v1" or not run["world"].get("walkways"):
            raise ValueError(f"Refusing a non-sidewalk replay: {scenario}")
        times = [float(frame["t"]) for frame in run["frames"]]
        if times != sorted(set(times)):
            raise ValueError(f"Frame times must be strictly increasing: {scenario}")
    return value


def muted(hex_color):
    """Make truth-only actors visibly dimmer against the dark map."""
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    bg = (19, 45, 54)
    return tuple(round(a*.35+b*.65) for a, b in zip(rgb, bg))


def dashed(draw, points, fill, width=1, dash=4, gap=4, closed=False):
    pts = list(points) + ([points[0]] if closed and points else [])
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if not length:
            continue
        for start in range(0, math.ceil(length), dash+gap):
            end = min(length, start+dash)
            draw.line((a[0]+dx*start/length, a[1]+dy*start/length,
                       a[0]+dx*end/length, a[1]+dy*end/length), fill=fill, width=width)


def rectangle_points(obj, xy, extra=0.):
    x, y = float(obj["x"]), float(obj["y"])
    length = float(obj.get("length", 4.5))+extra
    width = float(obj.get("width", 1.8))+extra
    a = float(obj.get("heading", 0.))
    cs, sn = math.cos(a), math.sin(a)
    return [xy(x+u*cs-v*sn, y+u*sn+v*cs)
            for u, v in [(-length/2, -width/2), (length/2, -width/2),
                         (length/2, width/2), (-length/2, width/2)]]


def inset(obj, fraction_length=.35, fraction_width=.70, offset=.12):
    a = float(obj.get("heading", 0.))
    return {**obj, "x": obj["x"]+obj.get("length", 4.5)*offset*math.cos(a),
            "y": obj["y"]+obj.get("length", 4.5)*offset*math.sin(a),
            "length": obj.get("length", 4.5)*fraction_length,
            "width": obj.get("width", 1.8)*fraction_width}


def actor(draw, obj, xy, scale, visible=True, main=False, contact=False, sim_time=0.):
    typ = str(obj.get("type", 1)).lower()
    x, y = xy(obj["x"], obj["y"])
    kind = obj.get("kind", "")
    pedestrian = typ in ("2", "person", "pedestrian") or kind == "pedestrian"
    cyclist = typ == "3" or kind == "cyclist"
    color = "#6f5c3f" if pedestrian else "#d9c28b" if cyclist else "#b7c7ce"
    if "status" in obj:
        color = robot_color(obj)
    if not visible:
        color = muted(color)
    robot = obj.get("kind") == "delivery_robot" or "status" in obj
    a = float(obj.get("heading", 0.))
    def local(u, v):
        return (x+u*math.cos(a)-v*math.sin(a), y-u*math.sin(a)-v*math.cos(a))
    if robot:
        # Minimum pixel size is explicitly disclosed; positions/headings and
        # backend geometric contact tests retain the saved .9 x .65 m body.
        glyph = {**obj, "length": max(float(obj.get("length", .9)), 10/scale),
                 "width": max(float(obj.get("width", .65)), 7/scale)}
        draw.polygon(rectangle_points(glyph, xy), fill=color, outline="#214d3b")
        for u in (-.32, 0., .32):
            for side in (-1, 1):
                wheel = {**glyph, "x": glyph["x"]+glyph["length"]*u*math.cos(a)-side*glyph["width"]*.52*math.sin(a),
                         "y": glyph["y"]+glyph["length"]*u*math.sin(a)+side*glyph["width"]*.52*math.cos(a),
                         "length": 2.4/scale, "width": 2.6/scale}
                draw.polygon(rectangle_points(wheel, xy), fill="#172f31")
        draw.polygon(rectangle_points(inset(glyph, .56, .55, -.02), xy), fill="#dce6bd", outline="#668b67")
        if main:
            pts = rectangle_points(glyph, xy, .6)
            draw.line(pts+[pts[0]], fill="#2b684b", width=1)
        if contact:
            pts = rectangle_points(glyph, xy, .55)
            draw.line(pts+[pts[0]], fill=COLORS["contact"], width=3)
        if not visible:
            dashed(draw, rectangle_points(glyph, xy, .6), "#b99591", width=1, dash=3, gap=3, closed=True)
        return
    if pedestrian:
        r = max(2.7, .38*scale)
        moving = math.hypot(float(obj.get("vx", 0.)), float(obj.get("vy", 0.))) > .05
        phase = math.sin(sim_time*9+float(obj.get("id", 0))*.71) if moving else 0.
        draw.line((local(-r*.25, -r*.32), local(-r*(1.15+phase*.4), -r*.65)), fill=color, width=2)
        draw.line((local(-r*.25, r*.32), local(-r*(1.15-phase*.4), r*.65)), fill=color, width=2)
        draw.line((local(0., -r*.5), local(phase*r*.5, -r)), fill=color, width=1)
        draw.line((local(0., r*.5), local(-phase*r*.5, r)), fill=color, width=1)
        draw.ellipse((x-r*.65, y-r*.65, x+r*.65, y+r*.65), fill=color)
        hx, hy = local(r*.35, 0.)
        draw.ellipse((hx-r*.44, hy-r*.44, hx+r*.44, hy+r*.44), fill="#e9d2a3" if visible else "#6e7766")
        if moving:
            draw.line((local(r*1.1, 0), local(r*2.5, 0)), fill="#387c56" if visible else "#40594b", width=1)
    elif cyclist:
        a = float(obj.get("heading", 0.))
        for d in (-.55, .55):
            cx, cy = xy(obj["x"]+math.cos(a)*d, obj["y"]+math.sin(a)*d)
            r = max(2, .31*scale)
            draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline=color, width=1)
        draw.line(rectangle_points(inset(obj, .65, .30, 0), xy)[:2], fill=color, width=2)
        draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
    else:
        draw.polygon(rectangle_points(obj, xy), fill=color)
        draw.polygon(rectangle_points(inset(obj), xy), fill="#395b64" if visible else "#28464e")
    if not visible:
        dashed(draw, rectangle_points(obj, xy, 1.2), "#b99591", width=1, dash=3, gap=3, closed=True)
    if main:
        draw.line(rectangle_points(obj, xy, 1.5)+[rectangle_points(obj, xy, 1.5)[0]], fill="#e1f2d8", width=1)
    if contact:
        points = rectangle_points(obj, xy, 1.3)
        draw.line(points+[points[0]], fill=COLORS["contact"], width=3)


def draw_panel(image, box, run, frame, title, fonts):
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=12, fill=COLORS["map"])
    draw.text((left+16, top+11), title, font=fonts[18], fill="#e1eee1")
    main = next((v for v in frame.get("vehicles", []) if str(v.get("id")) == "0"), None)
    score = None if not main or main.get("eligible") is False or main.get("done") else main.get("score")
    valid_score = isinstance(score, (int, float)) and math.isfinite(score)
    score_text = f"评分 {score:.3f}" if valid_score else "不评分"
    visible = main.get("visible_count", "—") if main else "—"
    hidden_count = main.get("hidden_count", "—") if main else "—"
    draw.text((left+16, top+39), f"机器人 #0  ·  {score_text}   |   可见 {visible} / 遮挡 {hidden_count}", font=fonts[12], fill="#a9cdb7")
    context = assistance_at(run, frame)
    if context:
        incident = context["incident"]
        reason = REASON_LABELS.get(incident.get("request_reason"), str(incident.get("request_reason") or "原因未提供"))
        draw.text((left+16, top+60), f"求助 #{incident['robot_id']}  ·  {reason}  ·  {incident['requested_at']:.1f}s", font=fonts[12], fill="#dcc99b")
        action_text = ACTION_LABELS.get(incident.get("action"), str(incident.get("action") or "方案未提供"))
        command = f"已于 {incident['finished_at']:.1f}s 下发：{action_text}" if context["issued"] else "模拟复核中，尚未下发指令" if context["started"] else "已进入队列，等待模拟席位"
        draw.text((left+16, top+81), command, font=fonts[12], fill="#9bcada")
        travel = context["travel_m"]
        travel_text = f"{travel:.2f}m" if isinstance(travel, (int, float)) and math.isfinite(travel) else "未提供"
        state = STATE_LABELS.get(context["state"], str(context["state"] or "执行状态未提供"))
        draw.text((left+16, top+102), f"{state}  |  本帧累计协助后行驶 {travel_text}", font=fonts[11], fill="#e0a08e" if context["state"] in ("onsite_required", "blocked_after_instruction") else "#b6cdb1")
    else:
        draw.text((left+16, top+62), "本帧尚无已发生的协助请求", font=fonts[12], fill="#9bb5a2")
        draw.text((left+16, top+84), "原模型评分与独立停滞兜底共同门控；不提前展示未来指令。", font=fonts[11], fill="#89a48d")
    world = run["world"]
    ww, wh = float(world.get("width", 100)), float(world.get("height", 70))
    scale = min((right-left-36)/ww, (bottom-top-141)/wh)
    ox = left+((right-left)-ww*scale)/2
    oy = top+130+((bottom-top-141)-wh*scale)/2
    xy = lambda x, y: (round(ox+float(x)*scale), round(oy+(wh-float(y))*scale))
    for x in range(0, int(ww)+1, 10):
        draw.line((xy(x, 0), xy(x, wh)), fill="#1e3b41", width=1)
    for y in range(0, int(wh)+1, 10):
        draw.line((xy(0, y), xy(ww, y)), fill="#1e3b41", width=1)
    for road in world.get("roads", []):
        points = [xy(*p) for p in road.get("points", [])]
        if len(points) >= 2:
            draw.line(points, fill="#4b665e", width=max(1, round((road.get("width", 8)+1.3)*scale)), joint="curve")
            draw.line(points, fill=COLORS["road"], width=max(1, round(road.get("width", 8)*scale)), joint="curve")
    for road in world.get("roads", []):
        points = [xy(*p) for p in road.get("points", [])]
        dashed(draw, points, "#60796b", dash=6, gap=7)
    for walkway in world.get("walkways", []):
        points = [xy(*p) for p in walkway.get("points", [])]
        if len(points) >= 2:
            draw.line(points, fill="#dfe6ca", width=max(1, round((walkway.get("width", 5)+.5)*scale)), joint="curve")
            draw.line(points, fill="#b7ccb1", width=max(1, round(walkway.get("width", 5)*scale)), joint="curve")
    for crossing in world.get("crosswalks", []):
        points = [xy(*p) for p in crossing.get("points", [])]
        if points:
            draw.polygon(points, fill="#e1e8d1")
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            # World crosswalks are axis-aligned; retain their saved extent.
            if max(xs)-min(xs) > max(ys)-min(ys):
                for x in range(min(xs)+2, max(xs), 5):
                    draw.line((x, min(ys), x, max(ys)), fill=COLORS["road"], width=2)
            else:
                for y in range(min(ys)+2, max(ys), 5):
                    draw.line((min(xs), y, max(xs), y), fill=COLORS["road"], width=2)
    for b in world.get("buildings", []):
        pts = rectangle_points(b, xy)
        draw.polygon([(x+3, y+4) for x, y in pts], fill="#0c232c")
        draw.polygon(pts, fill="#3a5857", outline="#648276", width=1)
        draw.polygon(rectangle_points(inset(b, .87, .82, 0), xy), fill="#334f51", outline="#4b6960")
        bx, by = xy(b["x"], b["y"])
        text = "建筑" if float(b.get("length", 10)) < 10 else str(b.get("label", "建筑"))
        building_font = fonts[10] if float(b.get("length", 10)) < 10 else fonts[11]
        bounds = draw.textbbox((0, 0), text, font=building_font)
        draw.text((bx-(bounds[2]-bounds[0])/2, by-8), text, font=building_font, fill="#b3cbc0")
    for route in world.get("routes", [])[:1]:
        points = route if isinstance(route, list) else route.get("points", [])
        pixels = [xy(*p) for p in points]
        dashed(draw, pixels, "#34764f", dash=3, gap=6)
        if pixels:
            x, y = pixels[-1]
            draw.ellipse((x-3, y-3, x+3, y+3), fill="#397d50")
    if context and context["issued"]:
        points = context["incident"].get("route_points")
        if isinstance(points, list) and len(points) > 1:
            dashed(draw, [xy(*p) for p in points], "#5cb4d8", width=2, dash=5, gap=3)
    ids = (main.get("visible_actor_ids") if main else None)
    if ids is None:
        ids = frame.get("visible_actor_ids")
    visible_ids = None if ids is None else {str(i) for i in ids}
    known_visible = lambda obj: visible_ids is None or str(obj.get("id")) in visible_ids
    for o in world.get("obstacles", []):
        kind = o.get("kind", "barrier")
        if kind == "parked_car":
            actor(draw, {**o, "type": 1}, xy, scale, visible=known_visible(o))
            continue
        draw.polygon(rectangle_points(o, xy), fill="#a57b4c" if kind != "trash" else "#6e7953", outline="#d1ac71", width=1)
        x, y = xy(o["x"], o["y"])
        if kind == "trash":
            for u, v, color in [(-.22, -.15, "#b8b482"), (.2, .12, "#8e9a6f"), (0, 0, "#d3c394")]:
                cx, cy = xy(o["x"]+o["length"]*u, o["y"]+o["width"]*v)
                rx, ry = max(2, o["length"]*scale*.16), max(2, o["width"]*scale*.22)
                draw.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), fill=color)
        else:
            # Draw stripe pieces inside the actual rotated barrier footprint.
            count = max(2, math.ceil(o.get("length", 4)/1.5))
            a = float(o.get("heading", 0.))
            for i in range(count):
                along = o.get("length", 4)*((i+.5)/count-.5)
                stripe = {**o, "x": o["x"]+math.cos(a)*along, "y": o["y"]+math.sin(a)*along,
                          "length": o.get("length", 4)/count*.45, "width": o.get("width", 2)*.85}
                draw.polygon(rectangle_points(stripe, xy), fill="#e1c78c")
        text = "垃圾堆" if kind == "trash" else "围挡"
        draw.text((x-16, y-16), text, font=fonts[10], fill="#dfc491")
    for a in frame.get("actors", []):
        actor(draw, a, xy, scale, visible=known_visible(a), sim_time=float(frame["t"]))
    contact_ids = {str(i) for i in frame.get("contact_vehicle_ids", [])}
    for v in frame.get("vehicles", []):
        if v.get("status") == "pending":
            continue
        is_main = str(v.get("id")) == "0"
        actor(draw, v, xy, scale, visible=is_main or known_visible(v), main=is_main,
              contact=str(v.get("id")) in contact_ids, sim_time=float(frame["t"]))
        if is_main or context and str(v.get("id")) == str(context["incident"]["robot_id"]):
            x, y = xy(v["x"], v["y"])
            draw.text((x-7, y-20), "#"+str(v["id"]), font=fonts[11], fill="#246045" if is_main else "#3a8cac")


def person_motion_audit(run):
    """Confirm actual same-ID pedestrian displacement in saved source frames."""
    moving_ids, moved_pairs, max_step = set(), 0, 0.
    for previous, current in zip(run["frames"], run["frames"][1:]):
        before = {str(a["id"]): a for a in previous.get("actors", []) if str(a.get("type")) == "2"}
        for person in current.get("actors", []):
            ident = str(person["id"])
            if str(person.get("type")) != "2" or ident not in before:
                continue
            other = before[ident]
            step = math.hypot(person["x"]-other["x"], person["y"]-other["y"])
            if step > 1e-8:
                moving_ids.add(ident)
                moved_pairs += 1
                max_step = max(max_step, step)
    return {"moving_pedestrian_ids": len(moving_ids), "same_id_displaced_frame_pairs": moved_pairs,
            "max_saved_step_m": max_step, "positions_reconstructed": False}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "demo/sidewalk_gallery.js")
    parser.add_argument("--output", type=Path, default=root / "demo/sidewalk_scenarios.gif")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--duration-ms", type=int, default=10000)
    parser.add_argument("--poster-time", type=float, default=70.)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    if not 2 <= args.frames <= 240 or args.duration_ms < 1000:
        parser.error("frames must be 2..240; duration-ms must be >=1000")
    if not args.font.is_file():
        parser.error("A local Chinese font is required; pass --font with its path")
    data = read_gallery(args.input)
    runs = {key: data["runs"][key] for key in SCENES}
    shared = sorted(set.intersection(*[{round(float(f["t"]), 8) for f in run["frames"]} for run in runs.values()]))
    if len(shared) < 2:
        raise ValueError("Scenarios must share at least two actual saved frame times")
    count = min(args.frames, len(shared))
    times = [shared[round(i*(len(shared)-1)/(count-1))] for i in range(count)]
    lookups = {key: {round(float(f["t"]), 8): f for f in run["frames"]} for key, run in runs.items()}
    fonts = {n: ImageFont.truetype(str(args.font), n) for n in (10, 11, 12, 13, 14, 18, 24)}
    panels = [(18, 76, 550, 486), (570, 76, 1102, 486), (18, 506, 550, 916), (570, 506, 1102, 916)]
    frames, poster = [], None
    command_indices = [i for i, t in enumerate(times) if any((assistance_at(runs[key], lookups[key][t]) or {}).get("issued") for key in SCENES)]
    candidates = command_indices or list(range(count))
    poster_index = min(candidates, key=lambda i: abs(times[i]-args.poster_time))
    for index, t in enumerate(times):
        image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["paper"])
        draw = ImageDraw.Draw(image)
        draw.text((22, 15), "移动人群中的机器人：求助与后续操作", font=fonts[24], fill=COLORS["ink"])
        draw.text((23, 49), "原模型评分 + 独立停滞兜底；模拟协助员按显式规则选择并下发指令。", font=fonts[13], fill=COLORS["muted"])
        time_text = f"模拟时间 {t:05.1f} s"
        draw.text((WIDTH-194, 25), time_text, font=fonts[18], fill=COLORS["ink"])
        for scenario, box in zip(SCENES, panels):
            draw_panel(image, box, runs[scenario], lookups[scenario][t], TITLES[scenario], fonts)
        draw.text((23, 936), "人 / 机器人：浅绿步道与斑马线；汽车（含停车）：灰色机动车道；机器人图标略放大（0.9 × 0.65米）。", font=fonts[13], fill=COLORS["ink"])
        draw.text((23, 960), "原道路 HistGB 是域外迁移评分；步道 / 避障 / 协助规则未做真实安全验证；分数不插值。", font=fonts[13], fill=COLORS["muted"])
        draw.text((23, 984), "蓝点线仅为已下发路径；下发只代表尝试，实际运动看本帧累计米数；失败、接触和未完成保留。", font=fonts[13], fill=COLORS["muted"])
        draw.text((23, 1008), "行人按真实保存坐标移动，摆腿仅辅助；GIF循环预览，120秒有限时域结束不代表已解卡。", font=fonts[12], fill=COLORS["muted"])
        if t == times[-1]:
            draw.text((WIDTH-377, 52), "回放时域已结束 · 未完成仍保留", font=fonts[12], fill="#a16b41")
        progress = index/(count-1)
        draw.rounded_rectangle((24, 1044, WIDTH-24, 1049), radius=2, fill="#d5e3d6")
        draw.rounded_rectangle((24, 1044, 24+max(1, (WIDTH-48)*progress), 1049), radius=2, fill="#39876a")
        if index == poster_index:
            poster = image.copy()
        frames.append(image.quantize(colors=192, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    # GIF delays use 10 ms units. Round to that granularity while keeping a
    # roughly ten-second loop; no synthetic trajectory interpolation is used.
    delay = max(10, round(args.duration_ms/count/10)*10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=delay,
                   loop=0, optimize=False, disposal=2)
    if poster is not None:
        poster.save(args.output.with_suffix(".png"))
    manifest = {"scenarios": list(SCENES), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "rendered_frames": count, "poster_saved_time_s": times[poster_index], "gif_loops_forever": True, "saved_frame_times_s": times, "gif_duration_ms": delay*count,
                "same_saved_time_in_all_panels": True, "model_recomputed": False,
                "scores_interpolated_or_imputed": False, "synthetic_motion": True, "domain": "sidewalk_robot", "robot_size_m": [.9, .65],
                "robot_icons_visually_enlarged": True,
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "command_timeline": {key: command_timeline(run) for key, run in runs.items()},
                "command_visibility_gating": "finished_at <= saved frame t; final incident.execution_state never used",
                "incident_selection_policy": "latest already-issued command by finished_at regardless of outcome; otherwise latest already-requested incident",
                "travel_evidence": "current saved robot.assisted_travel_m only; not final incident travel",
                "person_motion": {key: person_motion_audit(run) for key, run in runs.items()},
                "domain_counts": {key: {name: run.get("summary", {}).get(name) for name in ("off_walkway_frames", "cars_off_road_frames", "pedestrians_off_walkway_frames", "physical_overlap_frames")} for key, run in runs.items()},
                "contact_highlight": "saved contact_vehicle_ids; not an independent collision count",
                "bytes": args.output.stat().st_size}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scenarios": list(SCENES), "animated_frames": count,
                      "poster_saved_time_s": times[poster_index], "bytes": args.output.stat().st_size,
                      "source_sha256": manifest["source_sha256"],
                      "command_records": {key: len(value) for key, value in manifest["command_timeline"].items()},
                      "manifest": str(args.output.with_suffix(".manifest.json"))}, ensure_ascii=True))


if __name__ == "__main__":
    main()
