"""Render synthetic, genuinely model-scored trajectories as an animated GIF."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "demo/threshold_050_run.json")
    parser.add_argument("--output", type=Path, default=root / "demo/learned_simulation.gif")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8-sig"))
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if not font_path.exists():
        raise SystemExit("Set a local CJK font path in this rendering script; no network font download is performed.")
    fonts = {n: ImageFont.truetype(str(font_path), n) for n in (11, 13, 15, 18, 23)}
    colors = {"paper": "#eff4f0", "ink": "#17313a", "muted": "#72868b", "canvas": "#132d36", "lane": "#304a50", "green": "#93d5b2", "wait": "#ffc57a", "assist": "#83c6e6"}
    label = {"moving": "自主运行", "waiting": "自主等待", "queued": "协助排队", "assisting": "模拟协助中", "bypassing": "按模拟路线绕行", "done": "已到达"}
    frames = []
    indices = list(range(0, len(data["frames"]), 2))
    if indices[-1] != len(data["frames"])-1:
        indices.append(len(data["frames"])-1)
    for index in indices:
        frame = data["frames"][index]
        im = Image.new("RGB", (1000, 600), colors["paper"]); draw = ImageDraw.Draw(im)
        draw.text((25, 17), "学习模型驱动的配送交互模拟", font=fonts[23], fill=colors["ink"])
        draw.text((26, 53), f"真实 HistGB 分数 · 合成矩形场景 · 阈值 {data['config']['threshold']:.3f}（探索参数）", font=fonts[13], fill=colors["muted"])
        draw.rounded_rectangle((22, 89, 752, 439), radius=15, fill=colors["canvas"])
        scale = 6.8; ox = 47; oy = 406
        X = lambda x: int(ox+x*scale)
        Y = lambda y: int(oy-y*scale)
        for x in range(0, 101, 10):
            draw.line((X(x), Y(0), X(x), Y(40)), fill="#244049", width=1)
        for y in (8, 20, 32):
            draw.rounded_rectangle((X(0), Y(y+3.5), X(100), Y(y-3.5)), radius=3, fill=colors["lane"])
            for x in range(0, 100, 8):
                draw.line((X(x), Y(y), X(x+4), Y(y)), fill="#708a83", width=1)
        draw.line((X(92), Y(1), X(92), Y(39)), fill="#78bca2", width=2)
        draw.text((X(92)-12, Y(40)-15), "终点", font=fonts[11], fill=colors["green"])
        for obstacle in data["world"]["obstacles"]:
            x, y = X(obstacle["x"]), Y(obstacle["y"])
            l, w = obstacle["length"]*scale, obstacle["width"]*scale
            draw.rectangle((x-l/2, y-w/2, x+l/2, y+w/2), fill="#ae8659", outline="#dfb471", width=1)
            draw.text((x-24, y-29), "静态阻挡", font=fonts[11], fill="#dfb471")
        for actor in frame["actors"]:
            x, y = X(actor["x"]), Y(actor["y"])
            draw.ellipse((x-4, y-4, x+4, y+4), fill="#dbe1db")
        for v in frame["vehicles"]:
            x, y = X(v["x"]), Y(v["y"]); l, w = v["length"]*scale, v["width"]*scale
            color = colors["wait"] if v["status"] in ("waiting", "queued") else colors["assist"] if v["status"] == "assisting" else colors["green"]
            if v["done"]:
                color = "#6e9588"
            draw.rounded_rectangle((x-l/2, y-w/2, x+l/2, y+w/2), radius=3, fill=color)
            draw.rectangle((x, y-w*.35, x+l*.27, y+w*.35), fill="#2b5055")
            if v["id"] == 0:
                draw.ellipse((x-l*.8, y-w*1.2, x+l*.8, y+w*1.2), outline="#e7f4dd", width=1)
                draw.text((x-9, y-w/2-19), "#0", font=fonts[13], fill="#e7f4dd")
        draw.text((45, 414), "0 m", font=fonts[11], fill="#8caaa2")
        draw.text((684, 414), "100 m", font=fonts[11], fill="#8caaa2")
        draw.rounded_rectangle((768, 89, 978, 439), radius=15, fill="white")
        main = next(v for v in frame["vehicles"] if v["id"] == 0)
        draw.text((786, 108), f"模拟时间 {frame['t']:.1f} s", font=fonts[18], fill=colors["ink"])
        draw.text((786, 151), "跟踪车辆 #0", font=fonts[13], fill=colors["muted"])
        draw.text((786, 177), label.get(main["status"], main["status"]), font=fonts[18], fill=colors["ink"])
        score = main["score"] if main["eligible"] and not main["done"] else None
        draw.text((786, 219), "模型分数（非接管概率）", font=fonts[13], fill=colors["muted"])
        draw.text((786, 245), "—" if score is None else f"{score:.3f}", font=fonts[23], fill="#147e63")
        draw.rounded_rectangle((786, 287, 958, 295), radius=4, fill="#e4ece5")
        if score is not None:
            draw.rounded_rectangle((786, 287, 786+max(1,172*score), 295), radius=4, fill="#147e63")
        tx = 786+172*data["config"]["threshold"]
        draw.line((tx, 282, tx, 300), fill="#b27e32", width=2)
        draw.text((786, 321), f"协助队列  {len(frame['queue'])} 辆", font=fonts[15], fill=colors["ink"])
        occupied = sum(o["vehicle_id"] is not None for o in frame["operators"])
        draw.text((786, 354), f"席位占用  {occupied}/{len(frame['operators'])}", font=fonts[15], fill=colors["ink"])
        draw.text((786, 387), f"已到终点  {frame['completed']}/{data['config']['fleet']}", font=fonts[15], fill=colors["ink"])
        recent = [event for event in data["events"] if event["t"] <= frame["t"]]
        line = (f"车辆 #{recent[-1]['id']} · " + recent[-1]["message"]) if recent else "模型每秒使用过去1秒轨迹评分；运动规则独立于模型。"
        draw.text((26, 461), line, font=fonts[15], fill=colors["ink"])
        p = min(1, frame["t"]/120)
        draw.rounded_rectangle((26, 504, 974, 511), radius=3, fill="#dbe5df")
        draw.rounded_rectangle((26, 504, 26+max(1,948*p), 511), radius=3, fill="#147e63")
        draw.text((26, 532), "15秒协助、固定侧移绕行和停止净距均为模拟假设；没有验证真实配送控制或安全效果。", font=fonts[13], fill=colors["muted"])
        draw.text((26, 557), "默认开发阈值0.956的失败结果也保留在交互Demo中；本动画0.500用于展示参数变化。", font=fonts[13], fill=colors["muted"])
        frames.append(im)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=83, loop=0, optimize=True)
    frames[len(frames)//3].save(args.output.with_suffix(".png"))
    print(json.dumps({"animated_frames": len(frames), "bytes": args.output.stat().st_size, "synthetic": True}))


if __name__ == "__main__":
    main()
