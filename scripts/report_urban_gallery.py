"""Write an honest report from the completed synthetic urban run gallery."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gallery", type=Path, default=root/"demo/scene_gallery.js")
    p.add_argument("--results", type=Path, default=root/"results/urban")
    p.add_argument("--output", type=Path, default=root/"docs/urban_scenarios_zh.md")
    args = p.parse_args()
    gallery = json.loads(args.gallery.read_text(encoding="utf-8").split("=", 1)[1].strip().removesuffix(";"))
    manifest = json.loads((args.results/"manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(args.gallery.read_bytes()).hexdigest() != manifest["gallery_sha256"]:
        raise SystemExit("Gallery does not match completed run manifest")
    runs = gallery["runs"]
    lines = ["# 城市路口、建筑遮挡与多障碍动画实验", "",
        "2026-10-23追加：8类城市场景及遮挡开/关配对运行已经完成。评分仍由原开发集选中的全特征HistGradientBoosting给出，原真实数据、训练划分、算法及阈值未重新选择。以下都是合成情景，不是新的真实配送数据，也不是部署安全验证。", "",
        "## 打开与操作", "", "打开[交互页面](../demo/index.html)可离线切换八个预计算场景、播放、暂停、拖动时间及调整播放速度。需要重新计算不同阈值、车数、密度或遮挡设置时，启动本机服务：", "", "```text", "python scripts/serve_demo.py --samples artifacts/prepared/samples.csv --port 8766", "```", "",
        "随后打开 `http://127.0.0.1:8766/`。默认展示建筑盲角、6辆道路车辆、1名模拟协助员、种子42、密度1.0、原开发阈值0.95568571118795。可换成12/24辆、0.5/1.5倍密度、遮挡对照及机器人域外模式。预计算画廊每场120秒，帧间隔0.5秒；本地服务只在本机运行。", "",
        "## 场景与对象", "", "| 场景 | 内容 |", "| --- | --- |"]
    for key, run in runs.items():
        world = run["world"]
        lines.append(f"| {world.get('name', world.get('title', key))} | {world.get('description', '')} |")
    lines += ["", "![八类合成场景几何总览](../demo/urban_atlas.png)", "",
        "主车在弧线/折线道路上运动，车身朝向随路段改变；十字路口、T字路口与90度转角使用不同道路和路线，不能只是替换背景。行人、骑行者、来车、停放车辆、垃圾、围挡和建筑分别绘制。", "",
        "## 遮挡与算法边界", "",
        "建筑矩形实际遮挡视线。每辆配送车独立保留0.1秒间隔、过去一秒的观测历史；被楼遮住或不在范围内的对象不会进入模型，重新出现后也不会补填之前看不到的历史。界面可切换“主车观测/已知地图”和“全场真值”；真值视图中的淡虚线目标不表示模型已经看见它。", "",
        "原模型只支持车辆、行人、骑行者轨迹。**垃圾堆、围挡和建筑没有被伪装成车辆输入模型。** 已知建筑/道路地图、发现后的静态障碍、几何停止与协助后的栅格A*路径规划属于明确的规则层，不能归功于HistGB学会了避垃圾或绕建筑。规划可能失败，协助也不保证完成任务。", "",
        "模型每秒使用相同194列过去特征输出分数；当前已经进入两米代理阈值或没有当前可见目标时，不使用该分数申请协助。分数未经城市模拟域校准，不能解释为危险概率。道路车辆模式也有分布偏移，机器人模式偏移更明显。", "",
        "## 本次实际结果", "",
        "统一设置为6辆道路车、1名协助员、原开发阈值、密度1.0、种子42、建筑遮挡开启。只报告这一有限情景，不据此估计真实发生率。", "",
        "| 场景 | 完成/6 | 请求数 | 平均等待秒 | 遮挡目标车帧累计 | 矩形接触车帧 | 未支持静态物数 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, run in runs.items():
        s = run["summary"]; world = run["world"]
        lines.append(f"| {world.get('name', world.get('title', key))} | {s['completed']} | {s['requests']} | {s['mean_wait_s']:.2f} | {s['occluded_actor_frames']} | {s['physical_overlap_frames']} | {s['unsupported_static_objects']} |")
    lines += ["", "遮挡累计按每辆活动车、每0.5秒帧中被建筑遮住的受支持目标数求和。矩形接触按车辆/时间帧计数，同一持续接触会累计多次，**不是独立碰撞或事故次数**。未支持静态物包括建筑、垃圾和围挡。", "",
        "## 保留失败，不把脚本仿真当成安全证明", "",
        "本模拟的外生人车按预设轨迹运动，不会像现实交通参与者那样总是相互礼让；它们可能接触已经停下的配送车。接触会按全场真值统计并在画面中标红，未完成、排队和规划失败也会保留。不能把这些接触全部解释为学习模型主动撞上对象，更不能删除它们后宣称安全。", "",
        "模型只是近距离事件评分门控，不是转向、刹车、路权或协助需求模型。独立几何规则未建立真实刹车距离、轮胎动力学、法律路权、传感噪声或连续时间碰撞保证。建筑遮挡是二维中心视线近似，不是完整相机/激光雷达感知。", "",
        "## 遮挡对照与结论", "",
        "每个场景以相同种子、外生来流、模型和阈值关闭建筑遮挡再次运行，完整16行汇总见[场景指标](../results/urban/scenario_metrics.csv)。关闭遮挡只是改变可见输入的机制对照；不能预设它一定改变完成数或减少接触。单个种子的合成结果不提供真实环境的统计显著性证明。"]
    vis = args.results/"visibility_audit.json"
    if vis.exists():
        audit = json.loads(vis.read_text(encoding="utf-8"))
        angle = audit["occlusion_on"]["main_vehicle"]["maximum_heading_change_from_initial_deg"]
        lines += ["", f"[盲角独立遮挡核查](../results/urban/visibility_audit.json)确认目标路线为90度，但主车本次只转到{angle:.2f}度便停住，未完成拐弯；遮挡开/关两组完成数、请求数及接触数相同。遮挡确实改变了一部分观测和模型特征，但影响没有转化为本次结果改善，不能把路线箭头当作已经完成的轨迹。"]
    lines += ["", "这次扩展把场景覆盖推进到了转角、路口与未知静态障碍，但没有扩展原模型的训练类别。下一步若要研究真正的自主配送能力，需要配送域中的遮挡观测、受阻原因、可通行区域与协助标签，并做有相互反应的闭环交通验证。", "",
        "## 复现", "", "```text", "python scripts/build_urban_gallery.py --samples artifacts/prepared/samples.csv", "python scripts/render_urban_atlas.py", "python scripts/render_urban_animation.py", "python scripts/report_urban_gallery.py", "python -m unittest discover -s tests -v", "```", "",
        "原始数据、真实逐场景预测、特征缓存、账号与模型权重不进入公开包。画廊坐标均来自合成世界。项目仍待用户批准后才上传GitHub。", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": args.output.name, "scenarios": len(runs), "synthetic": True}))


if __name__ == "__main__":
    main()
