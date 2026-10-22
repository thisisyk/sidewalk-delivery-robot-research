"""Report actual sidewalk runs and their paired request-fallback comparison.

This is a simulator engineering evaluation, not another training experiment.
The report refuses a missing comparison or an unaudited gallery rather than
filling in prospective results.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


METRICS = ("completed", "unfinished", "requests", "physical_overlap_frames",
           "off_walkway_frames", "cars_off_road_frames",
           "pedestrians_off_walkway_frames")


def load_gallery(path):
    return json.loads(path.read_text(encoding="utf-8").split("=", 1)[1].strip().removesuffix(";"))


def totals(pairs, key):
    return {metric: sum(pair[key][metric] for pair in pairs) for metric in METRICS}


def section(path, heading, body):
    """Keep hand-written research prose, refresh just the engineering addendum."""
    start, end = "<!-- request-fallback:start -->", "<!-- request-fallback:end -->"
    block = f"{start}\n## {heading}\n\n{body}\n{end}"
    text = path.read_text(encoding="utf-8")
    if start in text:
        before, tail = text.split(start, 1)
        _, after = tail.split(end, 1)
        text = before + block + after
    else:
        first, rest = text.split("\n", 1)
        text = first + "\n\n" + block + "\n" + rest
    path.write_text(text, encoding="utf-8")


def actual_assistance_case(run):
    """Use the first crowd request, including a failed or unfinished outcome.

    Route installation is not evidence of movement. Movement is read from the
    controller's exported 0.1-second travel accumulator and actual saved poses.
    The case is selected by request time, never by a successful result.
    """
    lines = ["", "## 实际人工协助案例：人群场景", "",
        "本例固定选择人群场景最早发出的请求，不按是否成功筛选。这里的“人工”指规则模拟协助员；没有真人远程控制。"]
    incidents = run.get("assistance_incidents", [])
    if not incidents:
        return lines + ["", "本轮人群场景没有产生协助请求，因此没有可展示的实际指令或执行成功案例。"]
    incident = min(incidents, key=lambda x: (x["requested_at"], x["id"]))
    robot_id = incident["robot_id"]
    robot_requests = sorted((x for x in incidents if x["robot_id"] == robot_id), key=lambda x: x["requested_at"])
    trace = [(frame["t"], robot) for frame in run["frames"]
             for robot in frame["vehicles"] if robot["id"] == robot_id]
    assert trace, "an incident must have a saved robot trace"
    final_t, final_robot = trace[-1]
    def stamp(value):
        return "尚未发生" if value is None else f"{value:.2f}秒"
    score = incident.get("score_at_request")
    score_text = "无有效分数" if score is None else f"{score:.6f}"
    lines += ["", f"机器人ID为 `{robot_id}`，请求 `{incident['id']}`，原因 `{incident['request_reason']}`；"
        f"请求时模型资格为 `{incident['eligible_at_request']}`、评分为{score_text}。", "",
        "| 环节 | 实际时间或值 |", "| --- | --- |",
        f"| 提交请求 | {stamp(incident['requested_at'])} |",
        f"| 席位开始处理 | {stamp(incident.get('started_at'))} |",
        f"| 完成审查/下发指令 | {stamp(incident.get('finished_at'))} |",
        f"| 下发动作 | `{incident.get('action') or '尚未下发'}` |",
        f"| 指令来源 | `{incident.get('decision_source') or '尚未下发'}` |",
        f"| 本请求最后执行状态 | `{incident['execution_state']}` |"]
    instructions = incident.get("instructions", [])
    if instructions:
        lines += ["", "实际下发的操作步骤：", ""] + [f"{i}. {value}" for i, value in enumerate(instructions, 1)]
    if incident.get("explanation"):
        lines += ["", f"处理器当时的说明：{incident['explanation']}"]
    if incident.get("route_points"):
        lines += ["", "实际下发的路点（米，来自保存的指令；不重新生成路线）：", "", "```json",
                  json.dumps(incident["route_points"], ensure_ascii=False), "```"]
    else:
        lines += ["", "本请求没有下发新路点，不能据此声称已经成功绕行。"]
    lines += ["", "执行反馈按保存时间顺序列出：", "",
              "| 时间/秒 | 状态 | 实际记录 |", "| ---: | --- | --- |"]
    for event in incident.get("execution_events", []):
        lines.append(f"| {event['t']:.2f} | `{event['state']}` | {event['message']} |")
    finished_at = incident.get("finished_at")
    if finished_at is not None:
        next_request = next((x["requested_at"] for x in robot_requests if x["requested_at"] > incident["requested_at"]), final_t)
        before = [(t, r) for t, r in trace if t <= finished_at]
        after = [(t, r) for t, r in trace if finished_at <= t <= next_request]
        if before and after:
            base_t, base = before[-1]
            end_t, end = after[-1]
            if "assisted_travel_m" in base and "assisted_travel_m" in end:
                travelled = end["assisted_travel_m"] - base["assisted_travel_m"]
                displacement = math.hypot(end["x"]-base["x"], end["y"]-base["y"])
                lines += ["", f"在本指令下发前最后保存帧{base_t:.2f}秒至下一次请求前/模拟结束的{end_t:.2f}秒之间，"
                    f"控制器累计的指令后行程增加 **{travelled:.4f}米**，保存位置的净位移为 **{displacement:.4f}米**。"
                    "行程计数来自每0.1秒实际运动更新；窗口端点为0.5秒保存帧，不把路线下发或状态名称当作位移证据。"]
    lines += ["", f"截至{final_t:.2f}秒，该机器人任务终态为 `{final_robot['status']}`，"
        f"`done={final_robot['done']}`，位置为({final_robot['x']:.3f}, {final_robot['y']:.3f})米；"
        f"该机器人在本轮共请求{len(robot_requests)}次协助。"
        f"已下发指令后的累计实际行程为{final_robot.get('assisted_travel_m', '未提供')}米。", "",
        "`route_executing`表示路线已下发并进入执行尝试，单凭该状态不能证明脱困。是否实际移动、是否到达终点，分别由行程计数/保存坐标与 `done` 判断；"
        "若指令后继续受阻两秒，会记录 `blocked_after_instruction`，若重试则原请求记为 `replaced_by_retry`。失败不改写为协助成功。"]
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--gallery", type=Path, default=root / "demo/sidewalk_gallery.js")
    p.add_argument("--results", type=Path, default=root / "results/sidewalk")
    args = p.parse_args()
    gallery = load_gallery(args.gallery)
    audit = json.loads((args.results / "geometry_motion_audit.json").read_text(encoding="utf-8"))
    comparison = json.loads((args.results / "fallback_comparison.json").read_text(encoding="utf-8"))
    assert audit["status"] == "pass", "geometry and pedestrian motion must be audited first"
    assert audit["gallery_sha256"] == hashlib.sha256(args.gallery.read_bytes()).hexdigest(), "audit must match this gallery"
    assert comparison["status"] == "complete_paired_synthetic_comparison"
    pairs = comparison["pairs"]
    assert len(pairs) == 16, "report protocol is eight scenarios by two visibility settings"
    assert {(x["scenario"], bool(x["occlusion"])) for x in pairs} == {
        (scenario, occlusion) for scenario in gallery["runs"] for occlusion in (True, False)}
    baseline, hybrid = totals(pairs, "legacy"), totals(pairs, "hybrid")
    assert all(run["config"].get("fallback_enabled") is True for run in gallery["runs"].values())
    audit_rows = {row["scenario"]: row for row in audit["scenes"]}
    names = {key: run["world"]["name"] for key, run in gallery["runs"].items()}
    visible_pairs = {row["scenario"]: row for row in pairs if row["occlusion"]}
    assert set(visible_pairs) == set(names)
    for key, run in gallery["runs"].items():
        assert all(run["summary"][metric] == visible_pairs[key]["hybrid"][metric]
                   for metric in METRICS), "paired results must match the main gallery"
    first = next(iter(gallery["runs"].values()))
    fallback = first["metadata"].get("fallback_policy", {})
    assert fallback, "fallback thresholds and resource assumptions must be recorded"
    assistance = first["metadata"].get("assistance_policy", {})
    assert assistance, "the simulated operator assumptions must be recorded"
    text = ["# 人行道配送机器人：移动行人与独立求助机制", "",
        "当前主界面使用人行道配送机器人。**机器人和行人只在人行道与斑马线通行，全部汽车（包括停车）只在机动车道。** 行人有实际随时间变化的坐标；默认从人群场景首帧播放，可以暂停、重播、拖动时间或调整速度，不自动跳到协助或到达时刻。", "",
        "## 本轮修复与实验问题", "",
        "旧版把协助请求完全绑在近距模型的有效高分上。机器人已进入两米近距范围后，原任务资格无效，分数显示“—”；即使持续受阻也无法请求。协助后的重规划又依赖先提出请求，于是出现“无法评分→不请求→不重规划→继续堵塞”的闭环。空闲席位不能主动接到尚未进入队列的机器人。", "",
        "**本轮增加独立的行为求助通道：持续停滞、机器人互堵或多次路径规划失败，可以不依赖模型分数提出请求。** 模型的高分通道保留，几何约束和协助服务流程保留；同一请求记录原因。新增通道是人工设定的工程策略，不是模型从Waymo数据中学会了识别故障或解卡。", "",
        "本轮还把协助员处理改为显式的规则模块，记录指令与执行结果。因此主对照衡量的是“行为求助＋显式协助处理”的组合机制，不能把全部变化归因于请求触发或机器学习。若输出包含仅开启行为求助的消融，单独列出，避免混合归因。", "",
        "本轮问题是：在完全相同的场景、种子、人流、机器人和服务资源下，新增机制能否修复“受阻却不请求”，它又带来多少请求、等待或接触变化？不预设所有机器人都会到达。", "",
        "## 触发条件与工程假设", "",
        "原模型仍只接受过去约一秒观测，以同一194列特征和原开发集阈值0.95568571118795评分。当前已进入两米事件范围仍显示不评分；没有把无效分数改成高分来制造请求。模型门控通道仍要求至少两秒受阻。", "",
        "独立触发的实际配置记录如下，秒数、次数、冷却和重试上限均为本模拟的工程假设，没有用真实接管标签估计：", "", "```json",
        json.dumps(fallback, ensure_ascii=False, indent=2), "```", "",
        "协助采用先到先服务、每次15秒的简化假设。请求不等于已经获助，更不等于一定解困；队列积压、不可行路径和再次受阻均应保留。重规划只能使用当前已知静态障碍和许可步道/斑马线，不能为提高完成数穿越机动车道。", "",
        "## 人工协助模块与机器人执行", "",
        "`assistance.py`中的协助员是规则模拟器，没有真人远程操控或语言模型驾驶。机器人先提交原因和当时可观察状态，FIFO分配席位；服务完成后给出明确指令，机器人再按指令执行并记录反馈。协助方案记录如下：", "", "```json",
        json.dumps(assistance, ensure_ascii=False, indent=2), "```", "",
        "- `replan_walkway`：在已知障碍与许可通行域内给出新路线；本版还把**当前可见、速度小于0.05米/秒的其他机器人完整占地**作为临时规划障碍，绕开实际互堵位置，而不是再次批准穿过它的原路线。机器人安装路点后再由原几何约束逐步实走。保守规划器失败只表示没有找到可接受路径，不证明现实中必然无路。", "- `hold_then_resume`：对暂时性的动态阻挡先保持3秒，再按规则尝试恢复；不允许直接穿过行人或机器人。", "- `onsite_required`：没有可执行绕行方案时标记需要现场处理并保持停止，不伪造远程已经移除了垃圾、建筑或其他机器人。", "",
        "临时阻挡只使用此次审查时已观测到的静止机器人，不把移动或隐藏机器人冻结成已知地图障碍。规划后仍保留原逐步避障与通行域检查；静止目标之后可能移动，因此静态快照不等于未来通道保证。", "",
        "自动选择规则只看当前可见目标：4米内有移动行人且没有机器人阻挡时，优先短时让行；其余情况尝试许可区域内重规划。4米观察范围和3秒让行都只是显式工程假设。用户选择操作办法也不能授权越过步道边界或关闭避障。", "",
        "这些是离散的简化操作办法，不包含真实遥控、物体搬移或全面互相避让。获助后仍可能受阻，需查看后续运行、重试与最终状态，不能仅凭“协助结束”宣称成功。独立的[协助生命周期审计](../results/sidewalk/assistance_audit.json)核对保存的请求原因、时序、资源约束、指令与实际运动，不导入模拟控制器。", "",
        "## 配对设计与结果", "",
        "统一使用6个机器人、1名协助员、密度1.0、种子42及120秒。八场景各开/关建筑遮挡，共16个配对设置。旧版聚合快照保留在 `results/sidewalk_baseline/`；配对表见[完整CSV](../results/sidewalk/fallback_comparison.csv)和[JSON](../results/sidewalk/fallback_comparison.json)。每对保持外生轨迹和资源不变，主对照比较旧门控/处理器与行为求助/新协助模块的组合，另列仅改变求助通道的消融。", "",
        "以下将16个设置的机器人任务相加，仅描述这些固定合成运行；不是96条独立真实数据，不给出显著性或真实配送成功率：", "",
        "| 指标 | 旧模型门控 | 模型＋独立行为求助 |", "| --- | ---: | ---: |"]
    for metric, label in (("completed", "完成任务/96"), ("unfinished", "未完成任务/96"),
                          ("requests", "请求总次数"), ("physical_overlap_frames", "矩形接触机器人帧"),
                          ("off_walkway_frames", "机器人越界帧"), ("cars_off_road_frames", "汽车越界帧"),
                          ("pedestrians_off_walkway_frames", "行人越界帧")):
        text.append(f"| {label} | {baseline[metric]} | {hybrid[metric]} |")
    fallback_requests = sum(pair["hybrid"].get("fallback_requests", 0) for pair in pairs)
    text += ["", f"新运行中独立行为通道共记录{fallback_requests}次请求。相对旧版，完成任务变化为{hybrid['completed']-baseline['completed']:+d}，"
        f"矩形接触机器人帧变化为{hybrid['physical_overlap_frames']-baseline['physical_overlap_frames']:+d}，"
        f"仍有{hybrid['unfinished']}个任务未完成。请求恢复、任务完成与接触是不同结果，不能只选其中一个评价机制。"]
    reuse = comparison.get("counterfactual_reuse")
    if reuse:
        assert reuse["input_model_world_controller_unchanged"] is True
        assert reuse["only_primary_operator_helper_revised"] is True
        text += ["", "最终主策略的16个设置均在本版协助模块上重新运行。旧纯模型门控、仅行为求助/旧处理器、两秒等待请求和仅自主等待等控制分支，复用前一轮已经实际计算的结果：这些分支不调用新增协助模块，且生成器核对输入、模型、世界与控制器SHA-256均未改变，并核验来源比较文件、画廊与清单哈希。最初只处理静态地图的协助模块结果另保留为消融。", "",
            "因此这里是固定设置下的受控机制比较，不能把复用控制或相同场景的多策略运行算成80个独立新实验。复用证据记录在[比较JSON](../results/sidewalk/fallback_comparison.json)的 `counterfactual_reuse` 字段。"]
    text += ["", "16个配对设置的变化如下。等待为每机器人累计等待均值；接触按机器人/时间帧累计，不是独立事故次数。建筑遮挡开启的八个设置对应当前主回放。", "",
        "| 场景 | 建筑遮挡 | 完成：旧→新 | 请求：旧→新 | 平均等待/秒：旧→新 | 接触机器人帧：旧→新 |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in pairs:
        key = row["scenario"]
        a, b = row["legacy"], row["hybrid"]
        text.append(f"| {names[key]} | {'开' if row['occlusion'] else '关'} | {a['completed']}→{b['completed']} / 6 | {a['requests']}→{b['requests']} | {a['mean_wait_s']:.2f}→{b['mean_wait_s']:.2f} | {a['physical_overlap_frames']}→{b['physical_overlap_frames']} |")
    if all("fallback_only" in row for row in pairs):
        ablation = totals(pairs, "fallback_only")
        text += ["", "仅启用独立行为求助、保留旧协助处理器的消融如下。它帮助区分请求触发与协助指令改变的影响：", "",
            "| 策略 | 完成/96 | 请求总数 | 矩形接触机器人帧 |", "| --- | ---: | ---: | ---: |",
            f"| 旧纯模型门控＋旧处理器 | {baseline['completed']} | {baseline['requests']} | {baseline['physical_overlap_frames']} |",
            f"| 行为求助＋旧处理器 | {ablation['completed']} | {ablation['requests']} | {ablation['physical_overlap_frames']} |"]
        if all("static_only_operator" in row for row in pairs):
            static_only = totals(pairs, "static_only_operator")
            text.append(f"| 行为求助＋初版仅静态地图协助模块 | {static_only['completed']} | {static_only['requests']} | {static_only['physical_overlap_frames']} |")
        text.append(f"| 行为求助＋可见静止机器人绕行模块 | {hybrid['completed']} | {hybrid['requests']} | {hybrid['physical_overlap_frames']} |")
    if all("static_only_operator" in row for row in pairs):
        static_travel = sum(row["static_only_operator"]["travel_after_assistance_m"] for row in pairs)
        hybrid_travel = sum(row["hybrid"]["travel_after_assistance_m"] for row in pairs)
        text += ["", f"在具有相同指令后行程计数器的两版显式协助模块之间，初版累计指令后实际行程为{static_travel:.3f}米，"
            f"本版为{hybrid_travel:.3f}米。该计数只在指令下发后逐步累加实际运动，不把路点或执行尝试当作已经移动。"
            "初版规划仅看静态障碍，可能下发仍穿过静止机器人的路线；纳入当前可见阻挡是规划处理的修复，仍不是学习模型获得了新技能。"]
    text += ["", "## 求助原因与残留失败", "",
        "主回放的请求原因与未完成数如下。原因计数来自模拟输出，不由报告从模型分数推断。单次请求若有多个同时成立条件，以控制器实际记录的首要原因计数。", "",
        "| 场景 | 请求原因计数 | 请求总数 | 未完成 | 规划失败记录 |", "| --- | --- | ---: | ---: | ---: |"]
    for key, run in gallery["runs"].items():
        s = run["summary"]
        counts = s.get("requests_by_reason", {})
        text.append(f"| {names[key]} | `{json.dumps(counts, ensure_ascii=False, sort_keys=True)}` | {s['requests']} | {s['unfinished']} | {s.get('planning_failure_attempts', s.get('failed_replans', '未提供'))} |")
    incidents = [incident for run in gallery["runs"].values() for incident in run.get("assistance_incidents", [])]
    if incidents:
        actions, states = {}, {}
        for incident in incidents:
            action = incident.get("action") or "尚未下发"
            actions[action] = actions.get(action, 0) + 1
            state = incident["execution_state"]
            states[state] = states.get(state, 0) + 1
        text += ["", "主回放还保存每次请求的请求/开始/结束时刻、分数资格、下发动作、指令、路点和带时间戳的执行状态。它们可在界面展开核对，不以服务结束代替实际恢复。", "",
            f"八个主回放的指令计数：`{json.dumps(actions, ensure_ascii=False, sort_keys=True)}`。", "",
            f"请求在模拟结束时的执行状态计数：`{json.dumps(states, ensure_ascii=False, sort_keys=True)}`。"]
    text += actual_assistance_case(gallery["runs"]["crowd"])
    text += ["", "改动后，请求通道和任务完成要分别判断。若独立行为请求成功进入队列，只说明这条请求不再依赖模型资格；完成数、等待和接触的变化才说明后续简化协助规则产生了什么结果。**即便请求恢复，也不能据此声称所有死锁已解决、接触减少或模型预测变准。** 仍未完成的任务、接触帧和规划失败按输出保留。", "",
        "## 与其他规则策略的关系", "",
        "保存的每个主回放同时包含旧纯模型门控、等待两秒即请求、仅自主等待的汇总（`policies`）。这些策略用于机制对照，不能把新策略相对旧策略的变化全部归因于模型。请求次数也不能直接换算为人工时或每单成本；本模拟没有现实操作员效率数据。", "",
        "## 场景与空间约束", "",
        "![人行道多场景实际模型评分动画](../demo/sidewalk_scenarios.gif)", "",
        "| 场景 | 主要内容 |", "| --- | --- |"]
    for key, run in gallery["runs"].items():
        text.append(f"| {names[key]} | {run['world']['description']} |")
    text += ["", "![八类人行道场景几何总览](../demo/sidewalk_atlas.png)", "",
        "机器人0.9×0.65米完整矩形足迹必须被人行道/斑马线联合区域覆盖；转弯、绕垃圾和协助后的路径搜索也遵守该约束。汽车完整车体只在机动车道内，停放汽车没有上人行道的例外。垃圾、纸箱、围挡占用步道；建筑位于步道外侧并可遮挡视线。斑马线是许可过街位置，不意味着本模拟已经建立真实交通灯或法律路权。", "",
        "## 行人真实移动与独立检查", "",
        "每个场景对同一行人ID的连续帧坐标计算位移，并独立重算整个旋转矩形在许可区域内的覆盖面积。审计脚本不导入模拟控制器，既不只看中心点，也不用摆腿动画作为运动证据。详见[几何与移动审计](../results/sidewalk/geometry_motion_audit.json)。", "",
        "| 场景 | 移动行人轨迹数 | 最短保存行人位移/米 | 机器人越界帧 | 行人越界帧 | 汽车越界帧（动态/停车） |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for key in gallery["runs"]:
        a = audit_rows[key]
        text.append(f"| {names[key]} | {a['moving_pedestrian_tracks']} | {a['minimum_saved_pedestrian_distance_m']:.2f} | {a['robot_off_walkway']} | {a['pedestrian_off_walkway']} | {a['dynamic_car_off_road']} / {a['parked_car_off_road']} |")
    text += ["", "这些是保存帧的合成几何检查，不是连续时间或真实道路安全保证。行人可以同向、迎面、过街或从建筑后出现；原模型只接收当时实际看见并保存的历史，不补填此前被建筑遮住的位置。", "",
        "## 真实模型与规则的分工及结论边界", "",
        "分数仍由原开发集选择的全特征HistGradientBoosting产生，输入为同一194列过去轨迹特征。重训只使用原857个训练样本；原206例预测一致性记录保留。该模型来自道路车辆训练域，**没有训练过本次人行道配送机器人**，分数未在该域校准。其他机器人以车辆运动代理编码用于兼容特征，不代表模型已经学会机器人语义。建筑、垃圾与围挡没有被伪装成车辆。", "",
        "通行域约束、避让、路线跟随、栅格A*、独立行为求助、显式协助指令、15秒服务和排队均属于简化工程规则。外生行人和车辆采用脚本运动，没有全面双向反应、车辆动力学、传感误差或完整交通规则。结果只能说明这些规则在固定合成场景里的作用，不能证明真实配送接管率、安全性、模型性能或成本改善。", "",
        "本轮保留了原道路研究结果，没有新增真实评估样本、改标签、重新选模或宣称AP变化。它补足的是“无效模型分数时仍能发起行为求助”的系统逻辑；后续还需要独立机器人轨迹、停滞/接管标签及真实闭环验证。", "",
        "## 重新运行", "", "```text", "python scripts/build_sidewalk_gallery.py --samples artifacts/prepared/samples.csv", "python scripts/audit_sidewalk_gallery.py", "python scripts/audit_assistance_gallery.py", "python scripts/render_sidewalk_atlas.py", "python scripts/render_sidewalk_animation.py", "python scripts/report_sidewalk_gallery.py", "python scripts/serve_demo.py --samples artifacts/prepared/samples.csv --port 8766", "```", "",
        "本机服务可调整人流、车队规模、阈值及建筑遮挡后重新运行；离线页面仅切换已计算场景，不伪造参数重算。原始Waymo轨迹、真实逐场景预测、权重、缓存和账号不公开；用户已于2026-10-23授权公开本项目，授权范围与实际发布状态见[publication.json](../publication.json)。原运行和审计产物中的待审字段保留为生成时历史快照。", ""]
    (root / "docs/sidewalk_demo_zh.md").write_text("\n".join(text), encoding="utf-8")

    concise = (f"已加入不依赖模型评分的停滞、互堵和多次规划失败求助，并将协助处理改为显式指令模块。原模型高分通道保留；"
        f"分数无效不再自动阻断行为请求。固定八场景、开/关遮挡共16个配对设置中，"
        f"完成任务由{baseline['completed']}/96变为{hybrid['completed']}/96，"
        f"请求总数由{baseline['requests']}变为{hybrid['requests']}，"
        f"矩形接触机器人帧由{baseline['physical_overlap_frames']}变为{hybrid['physical_overlap_frames']}。"
        "这些是固定合成条件下两项改动的组合机制比较，不能全部归因于求助触发；新增通道不是机器学习训练结果；"
        "未完成、接触和规划失败全部保留。")
    for name in ("README.md", "docs/progress.md", "docs/validation.md", "docs/final_report_zh.md", "docs/demo_guide_zh.md"):
        path = root / name
        prefix = "docs/" if name == "README.md" else ""
        evidence = "results/sidewalk/" if name == "README.md" else "../results/sidewalk/"
        publication = "publication.json" if name == "README.md" else "../publication.json"
        body = (concise + f"\n\n完整触发假设、逐场景对照、原因计数和残留限制见[人行道报告]({prefix}sidewalk_demo_zh.md)"
            f"与[配对比较]({evidence}fallback_comparison.json)。原真实研究数据与模型选择不变；用户已于2026-10-23授权公开本项目，发布记录见[publication.json]({publication})。")
        section(path, "独立行为求助补充", body)
    print(json.dumps({"report": "sidewalk_demo_zh.md", "paired_settings": len(pairs),
                      "legacy_totals": baseline, "hybrid_totals": hybrid,
                      "spatial_and_motion_audit": "pass"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
