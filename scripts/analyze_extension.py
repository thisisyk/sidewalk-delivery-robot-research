"""Exploratory operating-point and subgroup audit of the frozen run.

Reads only previously frozen eval scores. It does not retrain, reselect a model,
change the target, or calibrate a threshold against evaluation labels. Outputs
are aggregate; identifiers and coordinates are never exported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from delivery_risk.experiment import apply_policy, budget_policy, selection_metrics

SCORES = {
    "constant_velocity": "constant_velocity",
    "hist_gradient_boosting": "full__hist_gradient_boosting__42",
    "random_forest": "full__random_forest__42",
}
PRESET_BUDGETS = (0.05, 0.10, 0.20)
PLOT_BUDGETS = tuple(i / 100 for i in range(5, 81, 5))
BOOTSTRAP_SEED = 40223


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(v) for v in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.bool_):
            return bool(item)
        return item
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def wilson(success, n):
    """Descriptive binomial interval; not a source-cluster confidence interval."""
    if not n or not float(success).is_integer():
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = success / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / (1 + z * z / n)
    return centre - half, centre + half


def subgroup_definitions(df):
    """Engineering cutpoints declared in source, never fitted to eval outcomes."""
    return [
        ("ego_speed_mps", "<2", df.ego_speed < 2),
        ("ego_speed_mps", "2_to_<8", (df.ego_speed >= 2) & (df.ego_speed < 8)),
        ("ego_speed_mps", ">=8", df.ego_speed >= 8),
        ("candidate_count", "1_to_4", df.neighbor_count <= 4),
        ("candidate_count", "5_to_9", (df.neighbor_count >= 5) & (df.neighbor_count <= 9)),
        ("candidate_count", "10_to_16", df.neighbor_count >= 10),
        ("current_clearance_m", ">2_to_<5", df.current_min_clearance < 5),
        ("current_clearance_m", "5_to_<10", (df.current_min_clearance >= 5) & (df.current_min_clearance < 10)),
        ("current_clearance_m", ">=10", df.current_min_clearance >= 10),
        ("pedestrian_present", "no", df.pedestrian_count == 0),
        ("pedestrian_present", "yes", df.pedestrian_count > 0),
        ("cyclist_present", "no", df.cyclist_count == 0),
        ("cyclist_present", "yes", df.cyclist_count > 0),
    ]


def cluster_intervals(df, draws):
    """Paired source-shard bootstrap; reapply same capacity rule in each draw."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = sorted(df.source_file.unique())
    index = {g: np.flatnonzero(df.source_file.to_numpy() == g) for g in groups}
    rows = []
    for budget in PRESET_BUDGETS:
        values = []
        for _ in range(draws):
            take = np.concatenate([index[g] for g in rng.choice(groups, size=len(groups), replace=True)])
            part = df.iloc[take]
            y = part.label_2m.to_numpy(int)
            if not y.sum():
                continue
            recalls = {}
            for name in ("hist_gradient_boosting", "constant_velocity"):
                scores = part[SCORES[name]].to_numpy(float)
                recalls[name] = selection_metrics(y, apply_policy(scores, budget_policy(scores, budget)))["recall"]
            values.append(recalls["hist_gradient_boosting"] - recalls["constant_velocity"])
        lo, hi = np.quantile(values, [.025, .975])
        rows.append({"budget": budget, "valid_draws": len(values), "ci95_low": lo, "ci95_high": hi})
    return {"method": "paired_source_shard_bootstrap", "source_groups": len(groups),
            "requested_draws": draws, "seed": BOOTSTRAP_SEED, "recall_difference": rows,
            "interpretation": "Exploratory pointwise intervals; five shards are not verified independent driving logs; no multiplicity adjustment."}


def plot_results(capacity, calibration, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"hist_gradient_boosting": "#087f8c", "constant_velocity": "#db7734", "random_forest": "#7064b2"}
    labels = {"hist_gradient_boosting": "HistGB (locked primary)", "constant_velocity": "Constant velocity", "random_forest": "Random forest"}
    fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.7))
    for model in SCORES:
        sub = capacity[capacity.model == model]
        axs[0].plot(sub.request_rate * 100, sub.recall * 100, marker="o", markersize=3,
                    color=colors[model], label=labels[model])
    axs[0].axvspan(0, 20, color="#087f8c", alpha=.055)
    axs[0].set(xlabel="Actual review capacity (% of eval cohort)", ylabel="Proxy-event recall (%)",
               xlim=(0, 81), ylim=(0, 102), title="Capacity changes the comparison")
    axs[0].legend(fontsize=8)
    for model in ("hist_gradient_boosting", "random_forest"):
        sub = calibration[(calibration.model == model) & (calibration.n > 0)]
        axs[1].plot(sub.mean_score, sub.event_rate, marker="o", color=colors[model], label=labels[model])
        for row in sub.itertuples():
            axs[1].annotate(f"n={row.n}", (row.mean_score, row.event_rate),
                            xytext=(2, 6), textcoords="offset points", fontsize=7)
    axs[1].plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
    axs[1].set(xlabel="Mean model score in fixed bin", ylabel="Observed proxy-event frequency",
               xlim=(-.02, 1.02), ylim=(-.02, 1.06), title="Scores are not guaranteed calibrated probabilities")
    axs[1].legend(fontsize=8, loc="lower right")
    for ax in axs:
        ax.grid(alpha=.18)
    fig.suptitle("Exploratory audit of frozen predictions | 206 scenes, 33 proxy events", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def percent(v):
    return f"{100 * v:.1f}%" if np.isfinite(v) else "无正例"


def make_report(capacity, transfer, subgroups, calibration, intervals):
    lines = ["# 冻结模型追加实验：复核容量、阈值迁移与失败分层", "",
        "本追加分析使用原实验已冻结的真实评估分数，未重新选择模型、调参或改变标签。主模型仍为全特征 HistGradientBoosting（seed=42）。这些是看到主结果后的探索性诊断，不能当作新的独立验证集。",
        "", "## 1. 收益取决于复核容量", "",
        "在同一个206例、33个未来三秒内新进入2米轮廓间距事件的评估队列中，按分数排序分配有限复核容量。容量为向下取整后的整数，边界同分采用与标签无关的随机化期望。", "",
        "| 名义容量 | 实际名额 | 恒速发现 / 33 | HistGB发现 / 33 | 随机森林发现 / 33 | HistGB－恒速召回差95%区间 |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for budget in PRESET_BUDGETS:
        part = capacity[np.isclose(capacity.budget, budget)].set_index("model")
        interval = next(r for r in intervals["recall_difference"] if r["budget"] == budget)
        lines.append(f"| {budget:.0%} | {int(part.loc['hist_gradient_boosting', 'capacity'])} | "
            f"{part.loc['constant_velocity', 'expected_true_positive']:.2f} | "
            f"{part.loc['hist_gradient_boosting', 'expected_true_positive']:.2f} | "
            f"{part.loc['random_forest', 'expected_true_positive']:.2f} | "
            f"[{interval['ci95_low'] * 100:.1f}, {interval['ci95_high'] * 100:.1f}] 个百分点 |")
    lines += ["", f"区间为{intervals['requested_draws']}次配对来源分片自举，只有{intervals['source_groups']}个评估分片，且未验证驾驶日志独立性；为逐点探索性区间，未做多重比较校正。区间不能代替更大规模外部验证。", "",
        "完整5%至80%容量曲线见下图。随着容量扩大，恒速方法也会覆盖更多事件；排序优势不能外推为所有运营负荷下的优势。", "",
        "![容量与分数可靠性](../results/extension/extension_diagnostics.png)", "",
        "## 2. 开发集阈值并不能固定线上负荷", "",
        "以下阈值均在原开发集确定；在评估集只应用、不重新设阈值。它与在评估队列强制选择前10%是两种策略。", "",
        "| 方法 | 开发集名义请求比例 | 评估请求数 | 评估请求比例 | 发现事件 | 事件召回 | 精确率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in transfer.itertuples():
        lines.append(f"| {row.model} | {row.nominal_dev_budget:.0%} | {row.expected_requests:.2f} | {percent(row.request_rate)} | {row.expected_true_positive:.2f} | {percent(row.recall)} | {percent(row.precision)} |")
    lines += ["", "固定名义10%开发集阈值后，HistGB在评估集只请求14/206例、发现13/33个事件；不能将它说成固定20个名额时的16/33。要保证真实远程协助工作量，需要额外的到达过程、处理时长、排队及部署分布研究。", "",
        "## 3. 失败模式：先固定切分，再报告全部组", "",
        "切点取自脚本中预先写定的工程阈值：速度2和8m/s、候选数4和9、当前轮廓间距5和10米、是否存在行人/骑行者。每个场景在不同维度中会重复出现；候选数受最多16个目标的截断。",
        "使用所有评估场景共享的开发集名义10%阈值；没有对子组重新挑选阈值。下表报告所有分组，不挑选表现最好的组。", "",
        "| 维度 / 分组 | 场景 | 事件 | HistGB TP / FN / FP | HistGB召回（描述性95%区间） | 恒速TP / FN / FP |",
        "| --- | ---: | ---: | --- | --- | --- |"]
    hist = subgroups[subgroups.model == "hist_gradient_boosting"]
    cv = subgroups[subgroups.model == "constant_velocity"].set_index(["dimension", "group"])
    for row in hist.itertuples():
        other = cv.loc[(row.dimension, row.group)]
        ci = f"[{percent(row.recall_ci95_low)}, {percent(row.recall_ci95_high)}]" if row.positive else "不可估计"
        lines.append(f"| {row.dimension} / {row.group} | {row.n} | {row.positive} | "
            f"{row.tp:.0f} / {row.fn:.0f} / {row.fp:.0f} | {percent(row.recall)} {ci} | "
            f"{other.tp:.0f} / {other.fn:.0f} / {other.fp:.0f} |")
    lines += ["", "召回区间仅为描述性的二项Wilson区间，忽略来源分片相关性和重复查看多个子组；不能用作显著性、群体公平性或优劣排名证据。尤其行人/骑行样本或正例很少的组，只能暴露证据空白。场景含某类对象不代表未来事件由该类对象造成。", "",
        "## 4. 分数不是已验证的运营风险概率", "",
        "使用固定分箱[0,.05)、[.05,.2)、[.2,.5)、[.5,.8)、[.8,1]检查分数与事件频率。未对这些评估分数进行再校准。CSV保留每箱样本量、均值和Wilson描述区间。空箱与稀疏箱不支持稳定的概率解释。",
        "", "| HistGB分数区间 | 场景 | 事件 | 平均模型分数 | 实际事件比例 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in calibration[calibration.model == "hist_gradient_boosting"].itertuples():
        right = "]" if row.bin_upper == 1 else ")"
        lines.append(f"| [{row.bin_lower:.2f}, {row.bin_upper:.2f}{right} | {row.n} | {row.positive} | {row.mean_score:.4f} | {percent(row.event_rate)} |")
    lines += ["", "例如，模型分数低于0.05的160例仍有8例事件，平均分数仅0.0048而事件频率为5.0%；最高分箱18例有15例事件，平均分数0.967而事件频率83.3%。这些小样本诊断说明，不能把模型分数直接读成准确的事件概率，更不能称为危险概率；低分也不等于安全。", "",
        "## 5. 比原结论新增了什么", "",
        "本次把“模型更好”具体化为“在有限容量和同一代理事件定义下，排序效率更高”，并揭示该收益随容量改变、固定开发阈值请求负荷会漂移、模型仍存在漏检，且若干对象子组证据很稀少。应把算法作为候选复核排序器，而不是自动刹车或真实人工接管判定器。",
        "", "本次仍是回放已有观测轨迹的离线分析，不是闭环驾驶。Waymo道路车辆不能直接代表文章中的低速人行道配送机器人；未来轨迹删失导致71.64%的原始场景排除、完整样本选择偏差与206例小评估集限制仍然存在。", "",
        "## 复现", "",
        "运行 `python scripts/analyze_extension.py --input <samples.csv> --predictions <predictions_2m.csv> --run <original_run_directory> --output results/extension --report docs/extension_report_zh.md`。输入需是本地受许可的原始实验产物，不随公开项目分发。输出仅为聚合指标；原始主实验目录保持不变。", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "extension")
    ap.add_argument("--report", type=Path, default=ROOT / "docs" / "extension_report_zh.md")
    ap.add_argument("--bootstrap-draws", type=int, default=1000)
    args = ap.parse_args()
    manifest_path = args.run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["input_sha256"] != sha256(args.input):
        raise ValueError("frozen_input_hash_mismatch")
    selection_path = args.run / "primary_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection["model"], selection["seed"], selection["ablation"], selection["threshold_m"]) != ("hist_gradient_boosting", 42, "full", 2):
        raise ValueError("unexpected_frozen_primary")
    for name, expected in manifest["code_sha256"].items():
        if sha256(ROOT / "src" / "delivery_risk" / name) != expected:
            raise ValueError(f"frozen_core_code_changed: {name}")
    df = pd.read_csv(args.input)
    predictions = pd.read_csv(args.predictions)
    cohort = df[(df.split == "eval") & (df.eligible_2m == 1)].copy()
    df = cohort.merge(predictions, on=["scenario_id", "source_file", "label_2m"],
                      validate="one_to_one", how="outer", indicator=True)
    if not (df._merge == "both").all() or len(df) != 206 or int(df.label_2m.sum()) != 33:
        raise ValueError("frozen_eval_cohort_mismatch")
    df = df.drop(columns="_merge")
    y = df.label_2m.to_numpy(int)
    capacity_rows = []
    for budget in PLOT_BUDGETS:
        for name, column in SCORES.items():
            score = df[column].to_numpy(float)
            policy = budget_policy(score, budget)
            capacity_rows.append({"model": name, "budget": budget, "capacity": policy["capacity"],
                                  **selection_metrics(y, apply_policy(score, policy))})
    capacity = pd.DataFrame(capacity_rows)
    policy_path = args.run / "dev_fixed_policies.csv"
    old_policies = pd.read_csv(policy_path)
    transfer_rows, subgroup_rows = [], []
    for name, column in SCORES.items():
        selected = old_policies[(old_policies.threshold_m == 2) & (old_policies.ablation == "full") &
                                (old_policies.seed == 42) & (old_policies.model == name)]
        for row in selected.itertuples():
            policy = {"cutoff": row.cutoff, "tie_probability": row.tie_probability}
            weights = apply_policy(df[column].to_numpy(float), policy)
            recomputed = selection_metrics(y, weights)
            for key, value in recomputed.items():
                if not np.isclose(value, getattr(row, key), rtol=1e-10, atol=1e-10):
                    raise ValueError(f"frozen_policy_metric_mismatch: {name} {key}")
            transfer_rows.append({"model": name, "nominal_dev_budget": row.nominal_dev_budget,
                                  "cutoff": row.cutoff, "tie_probability": row.tie_probability, **recomputed})
            if not np.isclose(row.nominal_dev_budget, .1):
                continue
            for dim, group, mask in subgroup_definitions(df):
                part = df[mask]
                sy = part.label_2m.to_numpy(int)
                sw = weights[np.asarray(mask)]
                metric = selection_metrics(sy, sw)
                tp = metric["expected_true_positive"]
                lo, hi = wilson(tp, int(sy.sum()))
                subgroup_rows.append({"model": name, "dimension": dim, "group": group,
                    "n": len(part), "source_shards": part.source_file.nunique(), "positive": int(sy.sum()),
                    "tp": tp, "fn": int(sy.sum()) - tp, "fp": metric["expected_requests"] - tp,
                    **metric, "recall_ci95_low": lo, "recall_ci95_high": hi,
                    "interval_caveat": "Descriptive Wilson only; ignores source-cluster dependence and multiplicity."})
    transfer = pd.DataFrame(transfer_rows)
    subgroups = pd.DataFrame(subgroup_rows)
    calibration_rows = []
    bins = (0, .05, .2, .5, .8, 1.00000000001)
    for name in ("hist_gradient_boosting", "random_forest"):
        score = df[SCORES[name]].to_numpy(float)
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (score >= lower) & (score < upper)
            n, positive = int(mask.sum()), int(y[mask].sum())
            lo, hi = wilson(positive, n)
            calibration_rows.append({"model": name, "bin_lower": lower, "bin_upper": min(1., upper),
                "n": n, "positive": positive, "mean_score": float(score[mask].mean()) if n else float("nan"),
                "event_rate": positive / n if n else float("nan"), "ci95_low": lo, "ci95_high": hi})
    calibration = pd.DataFrame(calibration_rows)
    intervals = cluster_intervals(df, args.bootstrap_draws)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("capacity_curve", capacity), ("fixed_dev_threshold_transfer", transfer),
                        ("subgroups_fixed_dev10", subgroups), ("score_reliability", calibration)):
        frame.to_csv(args.output / f"{name}.csv", index=False)
    write_json(args.output / "paired_capacity_bootstrap.json", intervals)
    plot_results(capacity, calibration, args.output / "extension_diagnostics.png")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(make_report(capacity, transfer, subgroups, calibration, intervals), encoding="utf-8")
    write_json(args.output / "extension_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete_exploratory_analysis",
        "model_retrained": False, "eval_used_to_select_model_or_threshold": False,
        "new_independent_evaluation_data": False, "n": len(df), "positive": int(y.sum()),
        "source_shards": int(df.source_file.nunique()), "primary_selection": selection,
        "input_sha256": sha256(args.input), "frozen_prediction_sha256": sha256(args.predictions),
        "original_run_manifest_sha256": sha256(manifest_path), "original_policy_sha256": sha256(policy_path),
        "script_sha256": sha256(__file__), "preset_capacity_budgets": PRESET_BUDGETS,
        "subgroup_cutpoints": {"speed_mps": [2, 8], "candidate_count": [4, 9], "current_clearance_m": [5, 10],
                               "object_presence": ["pedestrian", "cyclist"]},
        "publication_approval": "pending_user_review", "outputs_contain_identifiers_or_coordinates": False})
    print(json.dumps({"status": "complete", "n": len(df), "positive": int(y.sum()),
                      "subgroup_rows": len(subgroups), "bootstrap_draws": args.bootstrap_draws}, ensure_ascii=False))


if __name__ == "__main__":
    main()
