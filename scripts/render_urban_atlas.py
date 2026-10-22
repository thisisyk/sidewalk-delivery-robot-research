"""Render reproducible synthetic geometry diagrams; no model outcome is shown."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from delivery_risk.urban_worlds import SCENARIO_CATALOG, make_urban_world, urban_actors


COLORS = {"paper": "#eff3f7", "ink": "#203347", "muted": "#66768b",
          "ground": "#f7f8f8", "road": "#dee5ed", "edge": "#c2cdd8",
          "building": "#7b91aa", "trash": "#698c55", "barrier": "#ed9b43",
          "parked_car": "#7b828b", "car": "#4d7bc0", "pedestrian": "#ac4c95",
          "cyclist": "#c16e3d", "route": "#177f83", "secondary": "#9bb8c1"}

SUBTITLES = {
    "intersection": "直行道路交叉 / 斑马线行人 / 横向来车",
    "left_turn": "90° 圆弧转弯 / 对向车 / 转弯出口行人",
    "blind_corner": "绕街角建筑转弯 / 建筑遮挡侧面视线",
    "t_junction": "支路汇入 / 主路直行 / 车辆转弯",
    "trash": "大件与散落垃圾 / 停放车 / 部分车道阻塞",
    "construction": "两侧围挡形成窄口 / 施工车 / 出口行人",
    "crowd": "多方向行人 / 骑行者 / 路边停车",
    "mixed": "转弯 / 建筑 / 违停 / 垃圾 / 围挡 / 人车",
}


def box_points(item, project):
    result = []
    c, s = math.cos(item.get("heading", 0)), math.sin(item.get("heading", 0))
    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        dx, dy = a*item["length"]/2, b*item["width"]/2
        result.append(project(item["x"]+dx*c-dy*s, item["y"]+dx*s+dy*c))
    return result


def arrow(draw, points, fill, width=4, size=13):
    if len(points) < 2:
        return
    draw.line(points, fill=fill, width=width, joint="curve")
    a, b = points[-2:]
    heading = math.atan2(b[1]-a[1], b[0]-a[0])
    draw.polygon([b, (b[0]-size*math.cos(heading-.45), b[1]-size*math.sin(heading-.45)),
                  (b[0]-size*math.cos(heading+.45), b[1]-size*math.sin(heading+.45))], fill=fill)


def render(output, font_path):
    font = lambda size: ImageFont.truetype(str(font_path), size)
    f_title, f_sub, f_panel, f_note, f_small = [font(n) for n in (56, 28, 33, 25, 21)]
    width, outer, gap, header = 2200, 50, 34, 165
    panel_w, panel_h, row_gap = (width-outer*2-gap)//2, 790, 28
    footer_y = header+4*panel_h+3*row_gap+26
    image = Image.new("RGB", (width, footer_y+220), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw.text((outer, 28), "合成场景几何总览（非模型安全效果）", font=f_title, fill=COLORS["ink"])
    draw.text((outer+2, 101), "八种预设 · 地图 100 × 70 m · 动态对象取 t = 14 s · 箭头表示预设路线", font=f_sub, fill=COLORS["muted"])
    for index, scenario in enumerate(SCENARIO_CATALOG):
        col, row = index % 2, index // 2
        px, py = outer+col*(panel_w+gap), header+row*(panel_h+row_gap)
        draw.rounded_rectangle((px, py, px+panel_w, py+panel_h), radius=20, fill="white", outline="#d5dfe8", width=2)
        title = SCENARIO_CATALOG[scenario]["title"]
        draw.text((px+25, py+17), f"{index+1:02d}  {title}", font=f_panel, fill=COLORS["ink"])
        draw.text((px+27, py+62), SUBTITLES[scenario], font=f_note, fill=COLORS["muted"])
        mx, my, mw = px+24, py+103, panel_w-48
        scale = mw/100
        mh = scale*70
        canvas = Image.new("RGB", (round(mw), round(mh)), COLORS["ground"])
        cd = ImageDraw.Draw(canvas)
        project = lambda x, y: (round(x*scale), round(mh-y*scale))
        world = make_urban_world(scenario)
        for road in world["roads"]:
            points = [project(*p) for p in road["points"]]
            cd.line(points, fill=COLORS["edge"], width=round(road["width"]*scale)+4, joint="curve")
        for road in world["roads"]:
            points = [project(*p) for p in road["points"]]
            cd.line(points, fill=COLORS["road"], width=round(road["width"]*scale), joint="curve")
        for crosswalk in world["crosswalks"]:
            points = crosswalk["points"]
            x0, x1 = min(p[0] for p in points), max(p[0] for p in points)
            y0, y1 = min(p[1] for p in points), max(p[1] for p in points)
            if y1-y0 > x1-x0:
                y = y0
                while y < y1:
                    cd.polygon([project(x0, y), project(x1, y), project(x1, min(y+.7, y1)), project(x0, min(y+.7, y1))], fill="#ffffff")
                    y += 1.5
            else:
                x = x0
                while x < x1:
                    cd.polygon([project(x, y0), project(min(x+.7, x1), y0), project(min(x+.7, x1), y1), project(x, y1)], fill="#ffffff")
                    x += 1.5
        # Route arrows are geometry only, not a recorded model-driven trajectory.
        for route in world["routes"][1:]:
            arrow(cd, [project(*p) for p in route], COLORS["secondary"], 3, 13)
        arrow(cd, [project(*p) for p in world["routes"][0]], COLORS["route"], 6, 18)
        xstart, ystart = project(*world["routes"][0][0])
        cd.ellipse((xstart-7, ystart-7, xstart+7, ystart+7), fill=COLORS["route"], outline="white", width=2)
        for item in world["buildings"]:
            points = box_points(item, project)
            cd.polygon([(x+4, y+5) for x, y in points], fill="#cbd3dd")
            cd.polygon(points, fill=COLORS["building"], outline="#60798f")
            text = "建筑"
            x, y = project(item["x"], item["y"])
            bounds = cd.textbbox((0, 0), text, font=f_small)
            cd.text((x-(bounds[2]-bounds[0])/2, y-15), text, fill="white", font=f_small)
        for item in world["obstacles"]:
            points = box_points(item, project)
            cd.polygon(points, fill=COLORS[item["kind"]], outline="#46596b")
            x, y = project(item["x"], item["y"])
            if item["kind"] == "trash":
                r = max(4, round(min(item["length"], item["width"])*scale*.18))
                cd.ellipse((x-r-4, y-r, x+r-4, y+r), fill="#ab865c")
                cd.ellipse((x-r+5, y-r-4, x+r+5, y+r-4), fill="#577749")
            elif item["kind"] == "barrier":
                # White hatch marks keep orange construction visually distinct
                # from a car; no unsupported class is turned into model input.
                radius = min(item["width"], item["length"])*scale*.24
                cd.line((x-radius, y-radius, x+radius, y+radius), fill="#fff5dc", width=4)
            else:
                cd.line((x, y-5, x, y+5), fill="#cbd6e0", width=3)
        for actor in urban_actors(14., scenario):
            x, y = project(actor["x"], actor["y"])
            color = COLORS[actor["kind"]]
            if actor["type"] == 2:
                cd.ellipse((x-7, y-7, x+7, y+7), fill=color, outline="white", width=2)
                cd.line((x, y-11, x, y+11), fill=color, width=3)
            else:
                cd.polygon(box_points(actor, project), fill=color, outline="white", width=2)
                heading = actor["heading"]
                arrow(cd, [(x, y), (round(x+18*math.cos(heading)), round(y-18*math.sin(heading)))], "#314658", 2, 7)
        cd.rectangle((0, 0, round(mw)-1, round(mh)-1), outline="#dce3ea", width=2)
        image.paste(canvas, (round(mx), round(my)))
    draw.text((outer, footer_y), "图例", font=f_panel, fill=COLORS["ink"])
    entries = [("building", "建筑 / 视线遮挡"), ("trash", "垃圾堆 / 散落杂物"),
               ("barrier", "施工围挡"), ("parked_car", "停放车辆"), ("car", "动态车辆"),
               ("pedestrian", "行人"), ("cyclist", "骑行者"), ("route", "主车预设路线")]
    for i, (kind, text) in enumerate(entries):
        x = outer+110+(i % 4)*510
        y = footer_y+7+(i//4)*51
        draw.rounded_rectangle((x, y+1, x+31, y+28), radius=5, fill=COLORS[kind])
        draw.text((x+44, y-1), text, font=f_note, fill=COLORS["ink"])
    draw.text((outer, footer_y+122), "此图仅展示场景构成。路线与交通为显式合成假设；未展示预测分数、协助收益或安全结论。", font=f_sub, fill=COLORS["muted"])
    draw.text((outer, footer_y+167), "建筑、垃圾和围挡用于物理避障；原学习模型仅接收车辆、行人、骑行者及道路地图特征。", font=f_sub, fill=COLORS["muted"])
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)
    return image.size


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root/"demo/urban_atlas.png")
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    if not args.font.is_file():
        raise SystemExit("Provide --font with a local CJK TrueType/OpenType font.")
    size = render(args.output, args.font)
    print(f"Rendered synthetic geometry atlas: {args.output} ({size[0]} x {size[1]} px)")


if __name__ == "__main__":
    main()
