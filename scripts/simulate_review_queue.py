"""Conditional Monte Carlo of FIFO proxy-event review using frozen model scores.

This is an operations simulation, not closed-loop driving or measured human
intervention. No model is trained here and no threshold is tuned on evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 3600.0
DEADLINE = 120.0
RATES = (60, 120, 240, 480)
OPERATORS = (1, 2, 4)
SERVICE_SECONDS = (30, 60)
BASE_SEED = 20261023
POLICY_ORDER = ("all_fifo", "cv_dev10", "histgb_dev05", "histgb_dev10", "histgb_dev20")
LABELS = {"all_fifo": "Review all FIFO", "cv_dev10": "CV: dev 10% threshold",
          "histgb_dev05": "HistGB: dev 5% threshold", "histgb_dev10": "HistGB: dev 10% threshold",
          "histgb_dev20": "HistGB: dev 20% threshold"}
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.bool_):
            return bool(item)
        return item
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_policies(predictions, policy_file):
    original = pd.read_csv(policy_file)
    policies = {"all_fifo": {"model": "all", "cutoff": None, "tie_probability": 1.,
                              "nominal_dev_budget": None, "weights": np.ones(len(predictions))}}
    settings = (("cv_dev10", "constant_velocity", .10, "constant_velocity"),
                ("histgb_dev05", "hist_gradient_boosting", .05, "full__hist_gradient_boosting__42"),
                ("histgb_dev10", "hist_gradient_boosting", .10, "full__hist_gradient_boosting__42"),
                ("histgb_dev20", "hist_gradient_boosting", .20, "full__hist_gradient_boosting__42"))
    for name, model, budget, column in settings:
        rows = original[(original.threshold_m == 2) & (original.ablation == "full") &
                        (original.model == model) & (original.seed == 42) &
                        np.isclose(original.nominal_dev_budget, budget)]
        if len(rows) != 1:
            raise ValueError(f"ambiguous_original_dev_policy: {name}")
        row = rows.iloc[0]
        scores = predictions[column].to_numpy(float)
        weights = (scores > row.cutoff).astype(float) + (scores == row.cutoff) * row.tie_probability
        if not np.isclose(weights.sum(), row.expected_requests, atol=1e-10):
            raise ValueError(f"original_policy_weight_mismatch: {name}")
        if not np.isclose(weights @ predictions.label_2m.to_numpy(int), row.expected_true_positive, atol=1e-10):
            raise ValueError(f"original_policy_positive_mismatch: {name}")
        policies[name] = {"model": model, "score_column": column, "cutoff": float(row.cutoff),
                          "tie_probability": float(row.tie_probability), "nominal_dev_budget": budget,
                          "weights": weights}
    return policies


def poisson_stream(rate, rep, labels):
    """Common arrival/sample/tie stream across every policy, staffing and service."""
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, rate, rep]))
    arrivals = []
    t = 0.0
    while True:
        t += float(rng.exponential(HORIZON / rate))
        if t >= HORIZON:
            break
        arrivals.append(t)
    arrival = np.asarray(arrivals, dtype=float)
    choices = rng.integers(0, len(labels), size=len(arrival))
    ties = rng.random(len(arrival))
    return arrival, choices, ties, labels[choices]


def simulate_fifo(arrivals, selected, labels, operators, service):
    """Greedy identical-server schedule is FIFO for a sorted arrival stream."""
    arrivals, selected, labels = map(np.asarray, (arrivals, selected, labels))
    if len(arrivals) and (np.diff(arrivals) < 0).any():
        raise ValueError("arrival_order_must_be_nondecreasing")
    if not (len(arrivals) == len(selected) == len(labels)):
        raise ValueError("stream_size_mismatch")
    availability = [(0.0, i) for i in range(operators)]
    heapq.heapify(availability)
    completed_waits = []
    selected_count = completed = pending = in_service = 0
    selected_positive = completed_positive = timely_positive = late_completed_positive = pending_positive = 0
    busy_seconds = 0.0
    for arrival, accepted, positive in zip(arrivals, selected, labels):
        if not accepted:
            continue
        selected_count += 1
        selected_positive += int(positive)
        available, server = heapq.heappop(availability)
        start = max(float(arrival), available)
        finish = start + service
        heapq.heappush(availability, (finish, server))
        busy_seconds += max(0.0, min(finish, HORIZON) - min(start, HORIZON))
        if finish <= HORIZON:
            completed += 1
            completed_positive += int(positive)
            completed_waits.append(start - arrival)
            if finish - arrival <= DEADLINE:
                timely_positive += int(positive)
            else:
                late_completed_positive += int(positive)
        elif start < HORIZON:
            in_service += 1
            pending_positive += int(positive)
        else:
            pending += 1
            pending_positive += int(positive)
    candidate_positive = int(np.sum(labels))
    assert selected_count == completed + pending + in_service
    assert selected_positive == completed_positive + pending_positive
    assert completed_positive == timely_positive + late_completed_positive
    assert candidate_positive == (candidate_positive - selected_positive) + pending_positive + late_completed_positive + timely_positive
    assert 0 <= busy_seconds <= operators * HORIZON + 1e-8
    assert in_service <= operators
    return {"candidate_count": len(arrivals), "candidate_proxy_positive": candidate_positive,
            "selected_count": selected_count, "selected_proxy_positive": selected_positive,
            "completed_count": completed, "completed_proxy_positive": completed_positive,
            "pending_count": pending, "in_service_count": in_service,
            "unfinished_count": pending + in_service, "unfinished_proxy_positive": pending_positive,
            "unselected_proxy_positive": candidate_positive - selected_positive,
            "completed_late_proxy_positive": late_completed_positive,
            "timely_proxy_positive": timely_positive,
            "timely_proxy_recall": timely_positive / candidate_positive if candidate_positive else float("nan"),
            "service_utilization": busy_seconds / (operators * HORIZON),
            "busy_seconds": busy_seconds,
            "completed_only_p95_wait_seconds": float(np.quantile(completed_waits, .95)) if completed_waits else float("nan")}


def independent_fifo_check():
    """Small-event QA: event-driven reference, independent of greedy heap."""
    # At 3530, 3540, 3550: one 60-second server finishes first by the boundary,
    # leaves one in service and one waiting. A rejected event does not enqueue.
    arrivals = np.array([3530., 3540., 3550., 3560.])
    selected = np.array([True, True, True, False])
    labels = np.array([1, 0, 1, 1])
    actual = simulate_fifo(arrivals, selected, labels, 1, 60)
    expected = {"selected_count": 3, "completed_count": 1, "pending_count": 1,
                "in_service_count": 1, "timely_proxy_positive": 1, "unfinished_proxy_positive": 1,
                "busy_seconds": 70., "completed_only_p95_wait_seconds": 0.}
    for key, value in expected.items():
        assert np.isclose(actual[key], value), (key, actual[key], value)
    # General reference assigns chronological jobs to the earliest server using
    # a direct list; includes horizon censoring and both 30/60-second services.
    rng = np.random.default_rng(419)
    checked = 0
    for operators in OPERATORS:
        for service in SERVICE_SECONDS:
            arrivals = np.sort(rng.uniform(0, HORIZON, size=620))
            selected = rng.random(620) < .7
            labels = (rng.random(620) < .16).astype(int)
            available = [0.] * operators
            jobs = []
            for i in np.flatnonzero(selected):
                s = min(range(operators), key=lambda j: (available[j], j))
                start = max(arrivals[i], available[s])
                finish = start + service
                available[s] = finish
                jobs.append((start, finish, arrivals[i], labels[i]))
            complete = [j for j in jobs if j[1] <= HORIZON]
            current = [j for j in jobs if j[0] < HORIZON < j[1]]
            wait = [j for j in jobs if j[0] >= HORIZON]
            timely = sum(j[3] for j in complete if j[1] - j[2] <= DEADLINE)
            actual = simulate_fifo(arrivals, selected, labels, operators, service)
            reference = {"completed_count": len(complete), "in_service_count": len(current),
                         "pending_count": len(wait), "timely_proxy_positive": timely,
                         "busy_seconds": sum(max(0., min(j[1], HORIZON) - min(j[0], HORIZON)) for j in jobs),
                         "completed_only_p95_wait_seconds": np.quantile([j[0] - j[2] for j in complete], .95)}
            for key, value in reference.items():
                assert np.isclose(actual[key], value), (operators, service, key)
                checked += 1
    return {"boundary_case_passed": True, "independent_direct_server_reference_checks": checked,
            "queue_conservation_checked_every_replicate": True,
            "rho_recomputed_independently_for_every_condition": True}


def plot_metrics(metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"all_fifo": "#575f66", "cv_dev10": "#dc7b30", "histgb_dev05": "#69b4ad",
              "histgb_dev10": "#007f86", "histgb_dev20": "#516eb6"}
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 8.4), sharex=True, sharey=True)
    panels = ((1, 60), (2, 60), (4, 60), (1, 30))
    for ax, (operators, service) in zip(axes.flat, panels):
        for policy in POLICY_ORDER:
            sub = metrics[(metrics.operators == operators) & (metrics.service_seconds == service) &
                          (metrics.policy == policy)].sort_values("candidate_rate_per_hour")
            ax.plot(sub.candidate_rate_per_hour, sub.timely_proxy_recall_mean * 100,
                    color=colors[policy], marker="o", markersize=4, linewidth=1.8, label=LABELS[policy])
        ax.set_title(f"{operators} reviewer(s), {service} s per review", fontsize=11)
        ax.set(ylim=(-1, 101), xticks=RATES)
        ax.grid(alpha=.18)
    for ax in axes[-1]:
        ax.set_xlabel("Assumed candidate arrival rate (per hour)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Proxy positives reviewed within 120 s\nand before hour end (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3, fontsize=8)
    fig.suptitle("Post-event review queue using frozen learned scores\nTarget window 3 s; reviews 30/60 s: no preventive-intervention interpretation", fontsize=12)
    fig.text(.5, .014, "100 paired one-hour runs per condition. Assumed Poisson arrivals; conditional resampling of 206 scenes. Mean curves.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .135, 1, .94))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def report_text(metrics, policies, n_reps):
    lines = ["# 冻结算法分数驱动的人工复核排队模拟", "",
        "这次实验回答：当同一批候选场景以假设的速率到达，有限人员逐个复核时，模型筛选如何影响工作量和及时覆盖？它不是车辆控制实验，也不测量真实接管或运营成本。模型分数来自原先训练并在开发集选定的HistGB，未用手写风险分数替代，也未根据模拟结果改变模型或阈值。", "",
        "**关键时域限制：原标签只描述未来3秒内的代理事件，而这里单次复核需要30或60秒，完成期限设为120秒。最早的复核完成也已晚于标签的3秒窗口。因此全部队列指标只表示事后/离线复核资源规划，不能作为及时制动、实时干预、避免该事件或提升驾驶实时响应的证据。文中的“及时”只指人为设定的120秒复核期限。**", "",
        "## 固定假设", "",
        f"每次运行一小时；候选按齐次Poisson过程到达，速率为每小时60、120、240、480个。分别设置1、2、4名并行复核员，每次复核固定30或60秒，空队列开始。共24种条件，各{n_reps}个独立种子；每个种子在策略之间共享同一到达时间、同一场景抽样和同一边界同分随机数。",
        "每次到达从原先206例评估样本有放回抽样，其代理事件标签、恒速分数与冻结HistGB分数一起保留。这保留了分数与标签的经验关联，但假定了候选独立同分布、无时间相关和分布漂移；反复抽样不等于获得更多真实场景。",
        "请求进入无优先级FIFO队列，先到先服务，无取消、无抢占、无复核员差异或学习效应。主指标是全部到达代理正例中，在到达后120秒内且这一小时结束前完成事后复核的比例。小时结束仍排队/服务中的请求作为未完成保留，不把它们当作完成。",
        "这套30/60秒服务假设与动画演示中的简化服务时间分别定义，不应混合或互相验证。",
        "真实候选到达时间、接管需求和人工服务时长均未知；这里的数值均为人为设定的情景参数，不是对实际Uber Eats/Serve运营的估计。", "",
        "## 冻结筛选策略", "",
        "所有筛选阈值来自原开发集；评估集只用于固定阈值下的条件模拟。名义5%/10%/20%指开发集目标比例，在本评估样本中接受率不同。", "",
        "| 策略 | 固定评估样本接受数 / 206 | 接受率 | 可覆盖代理事件 / 33（排队前） |",
        "| --- | ---: | ---: | ---: |"]
    for name in POLICY_ORDER:
        p = policies[name]
        lines.append(f"| {LABELS[name]} | {p['eval_expected_accepted']:.2f} | {p['eval_acceptance_rate']:.2%} | {p['eval_expected_positive_accepted']:.2f} |")
    lines += ["", "HistGB开发集10%策略接受14/206例、覆盖13/33事件，恒速策略接受27/206例、覆盖14/33。前者工作量更低，同时筛选阶段少覆盖一个事件，不能说它在低负荷下覆盖率也更高。HistGB开发集5%策略仅覆盖5/33事件，即使精确率高也会漏掉大多数事件。", "",
        "## 全部24种条件的及时覆盖均值", "",
        "下表每格为全部到达代理正例的及时复核比例，不是对已筛选请求计算的精确率。2.5%/97.5%分位及所有工作量、期末积压、利用率和等待指标保存在CSV。", "",
        "| 每小时候选 | 复核员 | 服务秒 | 全部FIFO | 恒速dev10 | HistGB dev5 | HistGB dev10 | HistGB dev20 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for service in SERVICE_SECONDS:
        for operators in OPERATORS:
            for rate in RATES:
                sub = metrics[(metrics.service_seconds == service) & (metrics.operators == operators) &
                              (metrics.candidate_rate_per_hour == rate)].set_index("policy")
                values = " | ".join(f"{sub.loc[p, 'timely_proxy_recall_mean']:.1%}" for p in POLICY_ORDER)
                lines.append(f"| {rate} | {operators} | {service} | {values} |")
    lines += ["", "![排队模拟均值](../results/queue_simulation/figure.png)", "",
        "## 如何解释负荷与权衡", "",
        "解析负荷 rho = 候选每小时到达率 × 固定评估接受率 × 单次复核秒数 / (3600 × 复核员数)。rho描述假设长期到达/服务过程的期望负荷，单小时模拟仍包含起始空队列和结束截断效应。rho≥1提示该无限时域模型没有稳定余量；rho<1不保证120秒内完成事后复核。",
        "低流量或人力充足时，全部复核通常覆盖最多事件；筛选本身会丢掉未选事件。高流量且人力不足时，较少而较精确的请求可能减少积压、提高有限期限内的代理事件覆盖。这是资源与覆盖的权衡，不是机器学习在所有条件下全面优胜。",
        "较宽松的HistGB开发集20%阈值能在筛选阶段覆盖25/33事件，但也接受46/206候选；它在过载条件下可能积压。较严格阈值降低负荷同时降低覆盖上限，不能只展示低等待而忽略漏检。", "",
        "## 不确定性与等待时间的边界", "",
        f"所有区间都是{n_reps}次模拟结果的2.5%至97.5%经验分位范围，表示条件于当前206例经验分布以及Poisson/服务假设的Monte Carlo运行波动。它不是总体模型性能的置信区间，不包含样本选择、模型估计、真实分布漂移或人工时长不确定性。",
        "p95等待仅对这一小时内已完成的请求计算，遗漏积压中最慢的请求，具有条件幸存偏差。必须同时查看期末等待数、服务中数、未完成总数和及时覆盖率。服务利用率按观测小时内实际占用服务秒数/(人员×3600)计算，包含期末未完成服务已消耗的时间。",
        "即使请求在120秒内完成事后复核，也不能证明人工能避免三秒内的事件；该模型没有模拟复核后车辆动作、事故、真实干预或成本节省。代理事件只表示未来三秒内新进入2米轮廓间距。", "",
        "## 复现与核对", "",
        "运行 `python scripts/simulate_review_queue.py --predictions <predictions_2m.csv> --run <original_run_directory> --output results/queue_simulation --report docs/queue_simulation_zh.md`。该脚本读取私有冻结分数，公开产物仅含聚合指标。",
        "每次模拟核对 请求=完成+等待+服务中，以及代理正例分解守恒。独立直接服务器分配实现复核6组事件流与边界截断案例；解析rho使用固定接受率独立复算。原始模型、开发阈值和主实验指标不变。", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "queue_simulation")
    ap.add_argument("--report", type=Path, default=ROOT / "docs" / "queue_simulation_zh.md")
    ap.add_argument("--replicates", type=int, default=100)
    args = ap.parse_args()
    if args.replicates < 2:
        ap.error("at_least_two_replicates_required")
    pred = pd.read_csv(args.predictions)
    if len(pred) != 206 or int(pred.label_2m.sum()) != 33 or pred.scenario_id.duplicated().any():
        raise ValueError("unexpected_frozen_eval_cohort")
    selection = json.loads((args.run / "primary_selection.json").read_text(encoding="utf-8"))
    if (selection["model"], selection["seed"], selection["ablation"], selection["threshold_m"]) != ("hist_gradient_boosting", 42, "full", 2):
        raise ValueError("unexpected_primary_model")
    qa = independent_fifo_check()
    policies = make_policies(pred, args.run / "dev_fixed_policies.csv")
    labels = pred.label_2m.to_numpy(int)
    public_policies = {}
    for name, policy in policies.items():
        weights = policy["weights"]
        public_policies[name] = {key: value for key, value in policy.items() if key != "weights"}
        public_policies[name].update({"eval_expected_accepted": float(weights.sum()),
                                     "eval_acceptance_rate": float(weights.mean()),
                                     "eval_expected_positive_accepted": float(weights @ labels),
                                     "eval_positive_selection_recall": float(weights @ labels / labels.sum())})
    raw = []
    conservation_checks = 0
    for rate in RATES:
        for rep in range(args.replicates):
            arrival, choices, tie_uniform, sample_y = poisson_stream(rate, rep, labels)
            masks = {name: tie_uniform < policy["weights"][choices] for name, policy in policies.items()}
            for operators in OPERATORS:
                for service in SERVICE_SECONDS:
                    for name in POLICY_ORDER:
                        values = simulate_fifo(arrival, masks[name], sample_y, operators, service)
                        raw.append({"candidate_rate_per_hour": rate, "operators": operators,
                                    "service_seconds": service, "policy": name, "rep": rep, **values})
                        conservation_checks += 1
    frame = pd.DataFrame(raw)
    keys = ["candidate_rate_per_hour", "operators", "service_seconds", "policy"]
    columns = [col for col in frame.columns if col not in keys + ["rep"]]
    aggregate_rows = []
    for settings, part in frame.groupby(keys, sort=True):
        row = dict(zip(keys, settings))
        rate, operators, service, policy_name = settings
        acceptance = public_policies[policy_name]["eval_acceptance_rate"]
        rho = rate * acceptance * service / (3600 * operators)
        # Independent arithmetic using total expected accepted empirical counts.
        reference_rho = (rate / len(pred)) * public_policies[policy_name]["eval_expected_accepted"] / (operators * 3600 / service)
        assert np.isclose(rho, reference_rho)
        row.update({"replicates": len(part), "fixed_eval_acceptance_rate": acceptance,
                    "analytic_rho": rho})
        for col in columns:
            values = part[col].dropna().to_numpy(float)
            row[f"{col}_valid_replicates"] = len(values)
            row[f"{col}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{col}_p025"] = float(np.quantile(values, .025)) if len(values) else float("nan")
            row[f"{col}_p975"] = float(np.quantile(values, .975)) if len(values) else float("nan")
        aggregate_rows.append(row)
    metrics = pd.DataFrame(aggregate_rows)
    paired = []
    for (rate, operators, service), part in frame.groupby(keys[:-1], sort=True):
        pairs = part.pivot(index="rep", columns="policy", values="timely_proxy_recall")
        delta = (pairs.histgb_dev10 - pairs.cv_dev10).dropna().to_numpy(float)
        paired.append({"candidate_rate_per_hour": rate, "operators": operators, "service_seconds": service,
                       "histgb10_minus_cv10_timely_recall_mean": float(delta.mean()),
                       "p025": float(np.quantile(delta, .025)), "p975": float(np.quantile(delta, .975)),
                       "interpretation": "Paired simulation run spread, conditional on empirical sample and assumptions; not population confidence."})
    qa["simulated_queue_conservation_checks"] = conservation_checks
    qa["analytic_rho_checks"] = len(metrics)
    qa["status"] = "pass"
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    summary = {"status": "complete_conditional_monte_carlo", "source_scenes": 206, "source_proxy_positive": 33,
               "condition_count": len(RATES) * len(OPERATORS) * len(SERVICE_SECONDS),
               "policy_count": len(POLICY_ORDER), "replicates_per_condition": args.replicates,
               "total_policy_hour_runs": len(frame), "horizon_seconds": HORIZON,
               "completion_deadline_seconds": DEADLINE, "policies": public_policies,
               "target_prediction_window_seconds": 3,
               "temporal_limitation": "All completed reviews occur after the three-second target window because service is 30/60 seconds. Post-event/offline review planning only; no real-time preventive-intervention claim.",
               "paired_histgb10_vs_cv10": paired, "qa": qa,
               "uncertainty": "Conditional Monte Carlo empirical 2.5/97.5% run quantiles; not confidence intervals for true interventions or model population performance.",
               "no_real_arrival_or_human_service_measurements": True,
               "closed_loop_driving": False, "intervention_or_cost_effect_measured": False}
    write_json(args.output / "summary.json", summary)
    plot_metrics(metrics, args.output / "figure.png")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text(metrics, public_policies, args.replicates), encoding="utf-8")
    write_json(args.output / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(__file__), "frozen_prediction_sha256": sha256(args.predictions),
        "original_policy_sha256": sha256(args.run / "dev_fixed_policies.csv"),
        "primary_selection_sha256": sha256(args.run / "primary_selection.json"),
        "base_seed": BASE_SEED, "seed_derivation": "numpy SeedSequence([base_seed, rate, replicate]); identical streams across policies/staffing/service.",
        "candidate_rates_per_hour": RATES, "operators": OPERATORS, "service_seconds": SERVICE_SECONDS,
        "replicates": args.replicates, "queue_discipline": "FIFO; identical independent servers; no preemption; empty initial system",
        "target_prediction_window_seconds": 3,
        "temporal_limitation": "Review duration 30/60 seconds exceeds the three-second label window; every completed review is post-event for this target. The 120-second deadline is an assumed offline audit SLA, not a driving response deadline.",
        "animation_service_assumption_is_separate": True,
        "selection": "frozen development thresholds; common label-blind uniforms at score ties",
        "candidate_sampling": "with replacement from fixed 206 eval scores+labels, equal probability",
        "model_retrained_or_threshold_tuned": False, "raw_data_or_ids_exported": False,
        "publication_approval": "pending_user_review", "qa": qa,
        "outputs": {name: sha256(args.output / name) for name in ("metrics.csv", "summary.json", "figure.png")}})
    print(json.dumps({"status": "complete", "conditions": summary["condition_count"],
                      "aggregate_rows": len(metrics), "policy_runs": len(frame), "qa": qa}))


if __name__ == "__main__":
    main()
