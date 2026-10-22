"""Render a four-panel GIF from saved, genuinely model-scored urban replays.

This script never invokes the model or reconstructs actors. It uses only saved
frame coordinates, headings, eligibility, scores and contact/visibility fields.
The JavaScript input must be a JSON assignment, not executable JavaScript.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


SCENES = ("intersection", "blind_corner", "trash", "mixed")
TITLES = {"intersection": "十字路口 · 横向交通", "blind_corner": "盲角 · 建筑遮挡", "trash": "垃圾堆 · 停车绕行", "mixed": "综合街区 · 多重障碍"}
COLORS = {"paper": "#edf3ed", "ink": "#19383d", "muted": "#647d76", "map": "#132d36", "road": "#2c464b", "green": "#94d5b2", "wait": "#ffc57a", "assist": "#83c6e6", "contact": "#ff8b76"}
WIDTH, HEIGHT = 1120, 900


def read_gallery(path):
    text = path.read_text(encoding="utf-8-sig")
    prefix = re.match(r"\s*(?:window\.)?(?:DELIVERY_SCENE_GALLERY|DELIVERY_GALLERY)\s*=\s*", text)
    if not prefix:
        raise ValueError("Expected a JSON-only DELIVERY_SCENE_GALLERY assignment")
    value, end = json.JSONDecoder().raw_decode(text, prefix.end())
    if text[end:].strip() not in ("", ";"):
        raise ValueError("Executable content after gallery JSON is not accepted")
    for scenario in SCENES:
        run = value.get("runs", {}).get(scenario)
        if not run or not run.get("frames") or not run.get("world"):
            raise ValueError(f"Missing saved scenario replay: {scenario}")
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


def actor(draw, obj, xy, scale, visible=True, main=False, contact=False):
    typ = str(obj.get("type", 1)).lower()
    x, y = xy(obj["x"], obj["y"])
    kind = obj.get("kind", "")
    pedestrian = typ in ("2", "person", "pedestrian") or kind == "pedestrian"
    cyclist = typ == "3" or kind == "cyclist"
    color = "#e6dfc4" if pedestrian else "#d9c28b" if cyclist else "#b7c7ce"
    status = str(obj.get("status", ""))
    if "status" in obj:
        color = COLORS["wait"] if status in ("waiting", "queued", "blocked") else COLORS["assist"] if "assist" in status else COLORS["green"]
        if obj.get("done"):
            color = "#6b9283"
    if not visible:
        color = muted(color)
    if pedestrian:
        r = max(2.3, .38*scale)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
        angle = float(obj.get("heading", 0.))
        dx, dy = math.cos(angle)*r, -math.sin(angle)*r
        draw.line((x-dy*.5, y+dx*.5, x+dy*.5, y-dx*.5), fill=color, width=2)
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
    draw.text((left+16, top+39), f"主车 #0  ·  {score_text}   |   可见 {visible} / 遮挡 {hidden_count}", font=fonts[12], fill="#a9cdb7")
    world = run["world"]
    ww, wh = float(world.get("width", 100)), float(world.get("height", 70))
    scale = min((right-left-36)/ww, (bottom-top-77)/wh)
    ox = left+((right-left)-ww*scale)/2
    oy = top+66+((bottom-top-77)-wh*scale)/2
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
    for crossing in world.get("crosswalks", []):
        points = [xy(*p) for p in crossing.get("points", [])]
        if points:
            draw.polygon(points, fill="#768d7c")
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
        text = str(b.get("label", "建筑"))
        bounds = draw.textbbox((0, 0), text, font=fonts[11])
        draw.text((bx-(bounds[2]-bounds[0])/2, by-8), text, font=fonts[11], fill="#b3cbc0")
    for route in world.get("routes", [])[:1]:
        points = route if isinstance(route, list) else route.get("points", [])
        pixels = [xy(*p) for p in points]
        dashed(draw, pixels, "#9ebd8e", dash=3, gap=6)
        if pixels:
            x, y = pixels[-1]
            draw.ellipse((x-3, y-3, x+3, y+3), fill="#bbdfb3")
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
        actor(draw, a, xy, scale, visible=known_visible(a))
    contact_ids = {str(i) for i in frame.get("contact_vehicle_ids", [])}
    for v in frame.get("vehicles", []):
        if v.get("status") == "pending":
            continue
        is_main = str(v.get("id")) == "0"
        actor(draw, v, xy, scale, visible=is_main or known_visible(v), main=is_main,
              contact=str(v.get("id")) in contact_ids)
        if is_main:
            x, y = xy(v["x"], v["y"])
            draw.text((x-7, y-20), "#0", font=fonts[11], fill="#e4f2dc")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "demo/scene_gallery.js")
    parser.add_argument("--output", type=Path, default=root / "demo/urban_scenarios.gif")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--duration-ms", type=int, default=10000)
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
    panels = [(18, 76, 550, 421), (570, 76, 1102, 421), (18, 440, 550, 785), (570, 440, 1102, 785)]
    frames, poster = [], None
    poster_index = min(range(count), key=lambda i: abs(times[i]-20.))
    for index, t in enumerate(times):
        image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["paper"])
        draw = ImageDraw.Draw(image)
        draw.text((22, 15), "同一个学习模型，四种合成街区", font=fonts[24], fill=COLORS["ink"])
        draw.text((23, 49), "固定真实模型评分回放 · 道路车辆 · 红框保留当前矩形接触失败", font=fonts[13], fill=COLORS["muted"])
        time_text = f"模拟时间 {t:05.1f} s"
        draw.text((WIDTH-194, 25), time_text, font=fonts[18], fill=COLORS["ink"])
        for scenario, box in zip(SCENES, panels):
            draw_panel(image, box, runs[scenario], lookups[scenario][t], TITLES[scenario], fonts)
        draw.text((23, 802), "合成场景；评分来自原 HistGB；避障与路径规则未做安全验证；评分不是接管概率。", font=fonts[14], fill=COLORS["ink"])
        draw.text((23, 826), "淡显 / 虚线展示全场真值：目标可能被建筑遮挡或超出主车感知范围，未进入当前模型输入。", font=fonts[13], fill=COLORS["muted"])
        draw.text((23, 850), "建筑、垃圾堆与围挡不是原模型训练类别；完成任务不代表安全通过，动画未重算或补造分数。", font=fonts[13], fill=COLORS["muted"])
        progress = index/(count-1)
        draw.rounded_rectangle((24, 884, WIDTH-24, 889), radius=2, fill="#d5e3d6")
        draw.rounded_rectangle((24, 884, 24+max(1, (WIDTH-48)*progress), 889), radius=2, fill="#39876a")
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
                "rendered_frames": count, "saved_frame_times_s": times, "gif_duration_ms": delay*count,
                "same_saved_time_in_all_panels": True, "model_recomputed": False,
                "scores_interpolated_or_imputed": False, "synthetic_motion": True,
                "contact_highlight": "saved contact_vehicle_ids; not an independent collision count",
                "bytes": args.output.stat().st_size}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
