"""Render the separated motor-road / sidewalk / crosswalk simulation layouts."""
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from delivery_risk.sidewalk_worlds import SIDEWALK_CATALOG, make_sidewalk_world, sidewalk_actors
from render_urban_atlas import COLORS, arrow, box_points


def render(output, font_path):
    font = lambda size: ImageFont.truetype(str(font_path), size)
    title, sub, panel, small = [font(n) for n in (55, 27, 32, 22)]
    width, outer, gap, top, pw, ph = 2200, 50, 34, 172, 1033, 790
    footer = top+4*ph+3*28+26
    image = Image.new("RGB", (width, footer+225), "#edf3f1")
    draw = ImageDraw.Draw(image)
    draw.text((outer, 26), "人行道配送机器人 · 八种合成场景", font=title, fill=COLORS["ink"])
    draw.text((outer, 99), "机器人与人：仅人行道及斑马线  |  汽车（含停车）：仅机动车道  |  图示并非模型安全效果", font=sub, fill=COLORS["muted"])
    for index, scenario in enumerate(SIDEWALK_CATALOG):
        px, py = outer+(index % 2)*(pw+gap), top+(index//2)*(ph+28)
        draw.rounded_rectangle((px, py, px+pw, py+ph), 18, fill="white", outline="#d3e0dc", width=2)
        draw.text((px+24, py+17), f"{index+1:02d}  {SIDEWALK_CATALOG[scenario]['name']}", font=panel, fill=COLORS["ink"])
        descriptions = {
            "intersection": "沿北侧步道直行，经斑马线横穿支路",
            "left_turn": "沿街角步道 90° 转弯，不驶入机动车道",
            "blind_corner": "街角建筑遮挡步道上的迎面行人",
            "t_junction": "主路、支路、人行道及三条过街斑马线",
            "trash": "垃圾占步道；可经许可斑马线到对侧绕行",
            "construction": "围挡位于步道，施工车停在机动车道内",
            "crowd": "步道行人相遇，过街只走斑马线",
            "mixed": "纸箱、垃圾、围挡占步道；车只在车道",
        }
        draw.text((px+26, py+62), descriptions[scenario], font=sub, fill=COLORS["muted"])
        mw, scale = pw-48, (pw-48)/100
        mh = round(scale*70)
        canvas = Image.new("RGB", (mw, mh), "#f6f8f7")
        cd = ImageDraw.Draw(canvas)
        project = lambda x, y: (round(x*scale), round(mh-y*scale))
        world = make_sidewalk_world(scenario)
        for road in world["roads"]:
            cd.line([project(*p) for p in road["points"]], fill="#b0bac4", width=round(road["width"]*scale))
        for walkway in world["walkways"]:
            cd.line([project(*p) for p in walkway["points"]], fill="#b7d8bc", width=round(walkway["width"]*scale))
        for cross in world["crosswalks"]:
            points = cross["points"]
            x0, x1 = min(p[0] for p in points), max(p[0] for p in points)
            y0, y1 = min(p[1] for p in points), max(p[1] for p in points)
            cd.polygon([project(*p) for p in points], fill="#dde5de")
            if y1-y0 > x1-x0:
                for i in range(int((y1-y0)/1.5)+1):
                    y = y0+i*1.5
                    cd.polygon([project(x0, y), project(x1, y), project(x1, min(y+.7, y1)), project(x0, min(y+.7, y1))], fill="white")
            else:
                for i in range(int((x1-x0)/1.5)+1):
                    x = x0+i*1.5
                    cd.polygon([project(x, y0), project(min(x+.7, x1), y0), project(min(x+.7, x1), y1), project(x, y1)], fill="white")
        for route in world["routes"][1:]:
            arrow(cd, [project(*p) for p in route], "#83aa95", 2, 12)
        arrow(cd, [project(*p) for p in world["routes"][0]], "#0a6e55", 5, 17)
        for item in world["buildings"]:
            cd.polygon(box_points(item, project), fill="#7b91aa", outline="#667e94")
            x, y = project(item["x"], item["y"])
            cd.text((x-22, y-13), "建筑", font=small, fill="white")
        for item in world["obstacles"]:
            cd.polygon(box_points(item, project), fill=COLORS[item["kind"]], outline="#456052")
        for actor in sidewalk_actors(14., scenario):
            x, y = project(actor["x"], actor["y"])
            if actor["type"] == 1:
                cd.polygon(box_points(actor, project), fill=COLORS["car"], outline="white", width=2)
            else:
                cd.ellipse((x-6, y-6, x+6, y+6), fill=COLORS["pedestrian"], outline="white", width=2)
            direction = actor["heading"]
            arrow(cd, [(x, y), (round(x+18*math.cos(direction)), round(y-18*math.sin(direction)))],
                  "#5e3a62" if actor["type"] == 2 else "#36587d", 2, 7)
        # A recognizable robot at its actual route start, not an invented score
        # or a simulated progress point. It is labelled as the route start below.
        x, y = project(*world["routes"][0][0])
        cd.rounded_rectangle((x-8, y-6, x+8, y+6), 3, fill="#0a6e55", outline="white", width=2)
        cd.rectangle((x-5, y-9, x+5, y-6), fill="#244638")
        cd.rectangle((0, 0, mw-1, mh-1), outline="#d3e0dc", width=2)
        image.paste(canvas, (px+24, py+103))
    legend = [("#b0bac4", "机动车道"), ("#b7d8bc", "人行道"), ("#dde5de", "斑马线过街"), ("#0a6e55", "机器人预设路线"),
              (COLORS["building"], "建筑 / 视线遮挡"), (COLORS["trash"], "垃圾 / 纸箱"), (COLORS["barrier"], "施工围挡"), (COLORS["car"], "机动车道汽车")]
    for i, (color, label) in enumerate(legend):
        x, y = outer+(i % 4)*530, footer+(i//4)*48
        draw.rounded_rectangle((x, y+4, x+31, y+31), 4, fill=color)
        draw.text((x+43, y), label, font=sub, fill=COLORS["ink"])
    draw.text((outer, footer+111), "绿线为预设许可路线；小机器人在路线起点。人车取 t = 14 s，短箭头表示移动方向，没有构造模型分数。", font=sub, fill=COLORS["muted"])
    draw.text((outer, footer+157), "所有路线与交通均为合成；采用完整矩形面积检查通行域，最终实际运行轨迹另做审计。", font=sub, fill=COLORS["muted"])
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)
    return image.size


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root/"demo/sidewalk_atlas.png")
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    if not args.font.is_file():
        raise SystemExit("Provide a local CJK font with --font.")
    print(f"Rendered {args.output}, size={render(args.output, args.font)}")


if __name__ == "__main__":
    main()
