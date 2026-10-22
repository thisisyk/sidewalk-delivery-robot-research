"""Frozen CPU comparison; inputs are prepared real-data CSVs, never synthesized here.

Only aggregate outputs are public. Per-scene predictions stay in results/private.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .samples import Protocol, feature_columns


MODEL_NAMES = ("logistic_regression", "random_forest", "hist_gradient_boosting")
ABLATIONS = ("ego", "interaction", "full")
BUDGETS = (0.05, 0.10, 0.20)


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (42, 7, 2026)
    primary_seed: int = 42
    primary_threshold_m: int = 2
    thresholds_m: tuple[int, ...] = (2, 1, 3)
    bootstrap_draws: int = 300
    bootstrap_seed: int = 11421
    minimum_train: int = 200
    minimum_dev: int = 100
    minimum_eval: int = 100
    minimum_each_class: int = 10
    minimum_eval_shards: int = 5


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(frame: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    required = {"split", "scenario_id", "source_file", "source_sha256",
                "cv_min_clearance"}
    for threshold in cfg.thresholds_m:
        required.update((f"eligible_{threshold}m", f"label_{threshold}m"))
    absent = required - set(frame.columns)
    if absent:
        raise ValueError(f"missing_required_columns: {sorted(absent)}")
    if frame.empty:
        raise ValueError("no_prepared_samples")
    if set(frame["split"].unique()) != {"train", "dev", "eval"}:
        raise ValueError("all_and_only_train_dev_eval_splits_required")
    for name in ("scenario_id", "source_file", "source_sha256"):
        if frame[name].isna().any() or (frame[name].astype(str).str.len() == 0).any():
            raise ValueError(f"empty_metadata: {name}")
    if frame["scenario_id"].duplicated().any():
        raise ValueError("duplicate_scenario_ids: remove duplicated/overlapping IDs before training")
    if not frame["source_sha256"].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}").all():
        raise ValueError("source_sha256_must_be_64_hex_characters")
    hashes = frame["source_sha256"].astype(str).str.lower()
    if pd.DataFrame({"hash": hashes, "split": frame["split"]}).groupby("hash")["split"].nunique().gt(1).any():
        raise ValueError("same_source_hash_across_splits")
    if frame.groupby("source_file")["split"].nunique().gt(1).any():
        raise ValueError("same_source_file_across_splits")
    if frame.groupby("source_file")["source_sha256"].nunique().gt(1).any():
        raise ValueError("inconsistent_hash_for_source_file")
    trajectory_checked = "trajectory_sha256" in frame.columns
    if trajectory_checked:
        if not frame["trajectory_sha256"].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}").all():
            raise ValueError("invalid_trajectory_sha256")
        trajectories = frame["trajectory_sha256"].astype(str).str.lower()
        if pd.DataFrame({"trajectory": trajectories, "split": frame["split"]}).groupby("trajectory")["split"].nunique().gt(1).any():
            raise ValueError("same_reference_trajectory_across_splits")
    columns = feature_columns(frame.columns, "full")
    if not columns:
        raise ValueError("no_allowlisted_features")
    features = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if np.isinf(features).any():
        raise ValueError("infinite_input_features")
    # NaNs may be imputed using training statistics; future columns are not inputs.
    for threshold in cfg.thresholds_m:
        for column in (f"eligible_{threshold}m", f"label_{threshold}m"):
            values = pd.to_numeric(frame[column], errors="raise")
            if not values.isin([0, 1]).all():
                raise ValueError(f"nonbinary_or_missing_target: {column}")
    if not np.isfinite(pd.to_numeric(frame["cv_min_clearance"], errors="raise")).all():
        raise ValueError("invalid_cv_min_clearance")
    return {
        "rows": len(frame), "unique_scenarios": frame["scenario_id"].nunique(),
        "source_files": frame["source_file"].nunique(),
        "unique_source_hashes": hashes.nunique(), "full_feature_count": len(columns),
        "split_rows": frame["split"].value_counts().sort_index().to_dict(),
        "missing_feature_values": int(np.isnan(features).sum()),
        "source_hash_disjoint": True, "scenario_id_disjoint": True,
        "exact_reference_trajectory_overlap_checked": trajectory_checked,
        "log_id_disjoint_verified": False,
    }


def budget_policy(scores: np.ndarray, budget: float) -> dict:
    """Fixed integer capacity with fractional, label-blind randomization at ties."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or not len(scores):
        raise ValueError("finite_nonempty_scores_required")
    if not 0 <= budget <= 1:
        raise ValueError("budget_out_of_range")
    k = int(math.floor(budget * len(scores)))
    if k == 0:
        return {"cutoff": float("inf"), "tie_probability": 0.0, "capacity": 0}
    cutoff = float(np.partition(scores, len(scores) - k)[len(scores) - k])
    greater = int(np.sum(scores > cutoff))
    tied = int(np.sum(scores == cutoff))
    return {"cutoff": cutoff, "tie_probability": (k - greater) / tied,
            "capacity": k}


def apply_policy(scores: np.ndarray, policy: dict) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return (scores > policy["cutoff"]).astype(float) + (
        scores == policy["cutoff"]) * policy["tie_probability"]


def selection_metrics(labels: np.ndarray, weights: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    count = float(np.sum(weights))
    true_positive = float(np.dot(labels, weights))
    positives = int(labels.sum())
    rate = count / len(labels) if len(labels) else float("nan")
    recall = true_positive / positives if positives else float("nan")
    return {
        "expected_requests": count, "request_rate": rate,
        "expected_true_positive": true_positive,
        "recall": recall,
        "precision": true_positive / count if count else float("nan"),
        "recall_lift_vs_random": recall / rate if rate else float("nan"),
    }


def metric_row(labels: np.ndarray, scores: np.ndarray, probability: bool) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not len(labels) or len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("invalid_prediction_vector")
    positives = int(labels.sum())
    metrics = {
        "n": len(labels), "positive": positives, "negative": len(labels) - positives,
        "prevalence": float(labels.mean()),
        "ap": float(average_precision_score(labels, scores)) if positives else float("nan"),
        "roc_auc": float(roc_auc_score(labels, scores)) if 0 < positives < len(labels) else float("nan"),
        "brier": float(brier_score_loss(labels, scores)) if probability else float("nan"),
    }
    for budget in BUDGETS:
        selection = selection_metrics(labels, apply_policy(scores, budget_policy(scores, budget)))
        suffix = f"budget_{round(budget * 100):02d}pct"
        metrics.update({f"{key}_{suffix}": value for key, value in selection.items()})
    return metrics


def make_model(name: str, seed: int):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if name == "logistic_regression":
        return make_pipeline(imputer, StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
    if name == "random_forest":
        return make_pipeline(imputer, RandomForestClassifier(
            n_estimators=120, max_depth=12, min_samples_leaf=3,
            max_features="sqrt", n_jobs=1, random_state=seed))
    if name == "hist_gradient_boosting":
        return make_pipeline(imputer, HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, learning_rate=0.08,
            l2_regularization=0.1, early_stopping=False, random_state=seed))
    raise ValueError(f"unknown_model: {name}")


def cohort_counts(cohort: pd.DataFrame, label: str, cfg: ExperimentConfig) -> tuple[dict, bool]:
    counts = {}
    enough = True
    for split, minimum in (("train", cfg.minimum_train), ("dev", cfg.minimum_dev),
                           ("eval", cfg.minimum_eval)):
        y = cohort.loc[cohort["split"] == split, label].to_numpy(int)
        positive = int(y.sum())
        counts[split] = {"n": len(y), "positive": positive,
                         "negative": len(y) - positive}
        enough &= len(y) >= minimum and min(positive, len(y) - positive) >= cfg.minimum_each_class
    return counts, bool(enough)


def paired_bootstrap(labels: np.ndarray, score: np.ndarray, baseline: np.ndarray,
                     groups: np.ndarray, cfg: ExperimentConfig) -> dict:
    """Paired cluster bootstrap if enough shards; otherwise explicitly exploratory IID."""
    labels, score, baseline = map(np.asarray, (labels, score, baseline))
    unique_groups = np.unique(groups)
    cluster = len(unique_groups) >= 5
    rng = np.random.default_rng(cfg.bootstrap_seed)
    lookup = {g: np.flatnonzero(groups == g) for g in unique_groups}
    draws = []
    for _ in range(cfg.bootstrap_draws):
        if cluster:
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            indices = np.concatenate([lookup[g] for g in sampled])
        else:
            indices = rng.integers(0, len(labels), size=len(labels))
        y, a, b = labels[indices], score[indices], baseline[indices]
        if len(np.unique(y)) != 2:
            continue
        ap_delta = average_precision_score(y, a) - average_precision_score(y, b)
        recall_a = selection_metrics(y, apply_policy(a, budget_policy(a, .1)))["recall"]
        recall_b = selection_metrics(y, apply_policy(b, budget_policy(b, .1)))["recall"]
        draws.append((float(ap_delta), float(recall_a - recall_b)))
    arr = np.asarray(draws, dtype=float)
    result = {"draws_requested": cfg.bootstrap_draws, "valid_draws": len(draws),
              "unit": "source_shard" if cluster else "scenario_iid_exploratory",
              "source_shards": len(unique_groups), "original_log_groups_known": False,
              "caveat": "原始日志未知；区间不能排除重叠窗口相关性，小于五个分片时仅作探索。"}
    for index, name in enumerate(("ap_difference", "recall_at_10pct_difference")):
        result[name] = {"ci95_low": float(np.quantile(arr[:, index], .025)) if len(arr) else None,
                        "ci95_high": float(np.quantile(arr[:, index], .975)) if len(arr) else None}
    return result


def _plot_curves(path: Path, labels: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for name, scores in curves.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[0].plot(recall, precision, label=name)
        xs = np.linspace(.01, 1., 100)
        ys = [selection_metrics(labels, apply_policy(scores, budget_policy(scores, q)))["recall"]
              for q in xs]
        axes[1].plot(xs, ys, label=name)
    axes[0].axhline(float(np.mean(labels)), color="gray", linestyle=":", label="prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Primary 2 m proxy event")
    axes[1].plot([0, 1], [0, 1], "k:", label="random expected")
    axes[1].set(xlabel="Offline review budget", ylabel="Proxy-event recall",
                title="Tie-unbiased capacity ranking")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (float, int, np.number)) and np.isfinite(value) else "NA"


def _report(summary: dict, metrics: pd.DataFrame) -> str:
    primary = summary["primary_selection"]
    exploratory = summary["primary_status"] != "eligible_for_limited_empirical_comparison"
    provenance = summary.get("provenance", {})
    title = "# WOMD 实验自动结果草稿" if provenance.get("preparation_manifest_matches_input") else "# 输入来源未核实的实验草稿"
    lines = [title, "", f"状态：{summary['primary_status']}。",
             "本草稿由输入 CSV 的训练与评价自动生成。预处理来源清单与输入哈希是否匹配："
             f"{provenance.get('preparation_manifest_matches_input', False)}。来源真实性仍需结合官方下载记录核查。",
             "主模型在开发集确定；自动结果尚待人工案例审阅和用户批准，不能直接视为最终研究结论。", "",
             "## 主任务与样本", "",
             "使用过去一秒预测未来三秒新进入两米轮廓间距的代理事件。当前已接近场景排除；未来缺失整场景删失。",
             "近距离事件不等于危险、事故或真实接管需求。", "",
             "| 划分 | 场景 | 正例 | 阴例 |", "| --- | ---: | ---: | ---: |"]
    for split, count in summary["cohorts"].get("2m", {}).get("counts", {}).items():
        lines.append(f"| {split} | {count['n']} | {count['positive']} | {count['negative']} |")
    lines += ["", "## 锁定评估结果", "",
              f"开发集选择：`{primary.get('model', '不可选择')}`，固定种子 {primary.get('seed', 42)}，全特征。",
              "选择只比较三个全特征学习模型的开发集 AP；没有根据评估集更换主模型。", "",
              "| 模型 | AP | ROC-AUC | 10%预算召回 | 10%预算精确率 | Brier |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    if not metrics.empty:
        selected = metrics[(metrics.threshold_m == 2) & (metrics.split == "eval") &
                           (metrics.ablation == "full") & (metrics.seed == 42)]
        for _, row in selected.iterrows():
            lines.append(f"| {row['model']} | {_fmt(row['ap'])} | {_fmt(row['roc_auc'])} | "
                         f"{_fmt(row['recall_budget_10pct'])} | {_fmt(row['precision_budget_10pct'])} | {_fmt(row['brier'])} |")
    lines += ["", "## 自动结果解释（待人工审阅）", ""]
    comparison = summary.get("primary_comparison")
    if comparison:
        lines += [f"在本次保留样本上，所选模型相对恒速基线的 AP 差为 {_fmt(comparison['ap_difference'])}，"
                  f"10%复核预算的召回率差为 {_fmt(comparison['recall_at_10pct_difference'])}。"]
        uncertainty = summary.get("paired_bootstrap", {})
        for key, title in (("ap_difference", "AP差"), ("recall_at_10pct_difference", "10%预算召回率差")):
            ci = uncertainty.get(key, {})
            lines.append(f"{title}的配对自举95%区间：[{_fmt(ci.get('ci95_low'))}, {_fmt(ci.get('ci95_high'))}]。")
        lines.append(f"自举单位：`{uncertainty.get('unit', '未计算')}`；有效重复 {uncertainty.get('valid_draws', 0)} 次。")
        ci = uncertainty.get("ap_difference", {})
        low = ci.get("ci95_low")
        high = ci.get("ci95_high")
        if exploratory:
            reason = {"exploratory_below_minimum_counts": "主队列未达到预设样本或正负例数量",
                      "exploratory_insufficient_eval_shards": "评估集少于五个源分片，IID自举不能支持分片层面的稳定比较",
                      "exploratory_unverified_provenance": "缺少与输入匹配的官方数据预处理来源清单",
                      "exploratory_insufficient_bootstrap_draws": "可用自举重复次数不足"}.get(
                          summary["primary_status"], "未满足完整比较条件")
            lines.append(f"{reason}；以上仅为探索性数值，不构成稳定优越性结论。")
        elif low is not None and low > 0:
            lines.append("本样本的AP比较支持学习模型改善该代理任务的离线排序；此发现局限于已列数据与筛选条件。")
        elif high is not None and high < 0:
            lines.append("本样本的AP比较显示恒速基线更强，当前特征和模型未带来增益。")
        else:
            lines.append("AP差区间未给出稳定的单向优势；不能声称学习模型明确优于恒速基线。")
    else:
        lines.append("主任务无法完成有效比较；不生成实证性能结论。请查看状态与样本计数。")
    lines += ["", "## 其他分析与限制", "",
              "已完成的敏感性（1/2/3米）、三类特征消融和三个固定种子的数值见 metrics.csv；跳过的队列见 summary.json。"
              "种子重复没有被用于挑选最好结果；Logistic Regression的种子结果可能完全相同。",
              "dev_fixed_policies.csv 给出开发集确定的阈值策略在评估集的实际期望请求率，"
              "它可能偏离设定预算。边界平分采用与标签无关的随机复核期望。",
              "使用官方训练/验证边界和固定验证分片；原始日志ID未核实，不能宣称已严格排除所有重叠窗口。"
              "分片自举同样不能解决未知日志相关性。",
              ("输入清单标记为Uber Eats合作方Waymo的公开交通数据，仍需与官方下载记录核对。"
               if provenance.get("preparation_manifest_matches_input") else
               "输入来源未核实，不能作Uber Eats或Waymo的实证结论。") +
              "即便使用WOMD，也未确认任何场景属于外卖订单，且包含未标注的人工驾驶与自动驾驶模式。"
              "结果不能推断真实配送接管率、配送成本或事故降低。",
              "尚需人工检查有代表性的正确、误报与漏报案例；本自动报告不声明已经完成人工审阅。", "",
              "## 可重现性与发布", "",
              "运行参数与输入哈希见 run_manifest.json，特征清单见 feature_columns.json，"
              "模型选择时间记录见 primary_selection.json。逐场景预测只保存在 results/private。",
              "不上传原始数据、特征缓存、逐场景标签、预测或训练权重；用户审阅结论并明确批准后才可上传。", ""]
    stability = summary.get("seed_stability")
    if stability:
        lines += ["## 固定种子稳定性", "",
                  f"所选主模型全特征在三个固定种子的评估AP范围：{_fmt(stability['ap_min'])}—{_fmt(stability['ap_max'])}；"
                  f"10%预算召回率范围：{_fmt(stability['recall_10pct_min'])}—{_fmt(stability['recall_10pct_max'])}。",
                  "这些范围描述训练随机性，不能代替独立场景或日志上的抽样不确定性。", ""]
    return "\n".join(lines)


def run_experiment(input_csv: str | Path, output: str | Path,
                   cfg: ExperimentConfig | None = None) -> dict:
    cfg = cfg or ExperimentConfig()
    if cfg.thresholds_m != (2, 1, 3) or cfg.seeds != (42, 7, 2026):
        raise ValueError("frozen_protocol_requires_threshold_order_2_1_3_and_seeds_42_7_2026")
    if cfg.primary_seed not in cfg.seeds or cfg.primary_threshold_m != 2:
        raise ValueError("frozen_protocol_requires_seed_42_and_primary_2m")
    if cfg.primary_seed != 42 or cfg.bootstrap_draws < 1:
        raise ValueError("invalid_frozen_protocol_configuration")
    input_csv, output = Path(input_csv).resolve(), Path(output).resolve()
    if (output / "run_manifest.json").exists():
        raise FileExistsError("existing_run_manifest: choose a fresh --output directory to prevent stale results")
    output.mkdir(parents=True, exist_ok=True)
    private = output.parent / "private" / output.name
    private.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(input_csv)
    provenance_path = input_csv.parent / "provenance.json"
    provenance = {"preparation_manifest_matches_input": False,
                  "official_download_independently_verified": False,
                  "provenance_sha256": None}
    if provenance_path.exists():
        prepared = json.loads(provenance_path.read_text(encoding="utf-8"))
        if prepared.get("samples_sha256") != digest:
            raise ValueError("input_hash_mismatch_with_preparation_provenance")
        if prepared.get("protocol_sha256") != Protocol().fingerprint():
            raise ValueError("preparation_protocol_fingerprint_mismatch")
        provenance.update(provenance_sha256=sha256_file(provenance_path),
            preparation_manifest_matches_input=prepared.get("synthetic") is False and
                prepared.get("source_type") == "official_womd_scenario")
    frame = pd.read_csv(input_csv, dtype={"scenario_id": str, "source_file": str,
                                        "source_sha256": str})
    audit = preflight(frame, cfg)
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "input_sha256": digest, "input_filename": input_csv.name,
                "config": asdict(cfg), "preflight": audit,
                "provenance": provenance,
                "model_parameters": {name: {step: estimator.get_params(deep=False)
                    for step, estimator in make_model(name, 42).named_steps.items()}
                    for name in MODEL_NAMES},
                "code_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                for name in ("experiment.py", "samples.py", "geometry.py")},
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
                "approval_status": "pending_user_review", "models_saved": False,
                "publication_ready": False, "manual_case_review": "pending",
                "data_authenticity_note": "脚本不合成数据；来源以提取清单和官方文件SHA256审计为准。"}
    _write_json(output / "run_manifest.json", manifest)
    features = {name: feature_columns(frame.columns, name) for name in ABLATIONS}
    _write_json(output / "feature_columns.json", features)
    summary = {"primary_status": "not_evaluated", "primary_selection": {}, "cohorts": {},
               "provenance": provenance, "publication_ready": False,
               "manual_case_review": "pending", "user_approval": "pending"}
    rows, policy_rows = [], []
    primary_scores = {}
    primary_eval = None
    # Threshold 2 comes first. Its choice is frozen before any held-out predictions.
    for threshold in cfg.thresholds_m:
        label = f"label_{threshold}m"
        cohort = frame.loc[frame[f"eligible_{threshold}m"].astype(int) == 1].copy()
        parts = {s: cohort.loc[cohort.split == s].copy() for s in ("train", "dev", "eval")}
        counts, enough = cohort_counts(cohort, label, cfg)
        eval_shards = parts["eval"]["source_file"].nunique()
        summary["cohorts"][f"{threshold}m"] = {"counts": counts,
            "minimum_counts_met": enough, "eval_source_shards": eval_shards,
            "minimum_eval_shards_met": eval_shards >= cfg.minimum_eval_shards,
            "status": "pending"}
        if any(len(p) == 0 for p in parts.values()) or parts["train"][label].nunique() < 2 or parts["dev"][label].sum() == 0:
            summary["cohorts"][f"{threshold}m"]["status"] = "skipped_insufficient_classes_or_empty_split"
            if threshold == 2:
                summary["primary_status"] = "insufficient_classes_no_empirical_conclusion"
            continue
        y = {s: p[label].to_numpy(int) for s, p in parts.items()}
        trained = {}
        dev_candidates = []
        # Selection uses only full-feature seed 42, with complexity order for exact ties.
        for name in MODEL_NAMES:
            model = make_model(name, 42)
            model.fit(parts["train"][features["full"]], y["train"])
            dev_score = model.predict_proba(parts["dev"][features["full"]])[:, 1]
            trained[("full", name, 42)] = (model, dev_score)
            dev_candidates.append((float(average_precision_score(y["dev"], dev_score)), name))
        if threshold == 2:
            best_ap = max(value for value, _ in dev_candidates)
            chosen = next(name for value, name in dev_candidates if value == best_ap)
            selection = {"model": chosen, "seed": 42, "ablation": "full", "threshold_m": 2,
                         "selection_metric": "dev_ap", "dev_ap": best_ap,
                         "candidate_dev_ap": dict((name, ap) for ap, name in dev_candidates),
                         "frozen_before_eval_prediction": True,
                         "frozen_utc": datetime.now(timezone.utc).isoformat()}
            summary["primary_selection"] = selection
            if not enough:
                summary["primary_status"] = "exploratory_below_minimum_counts"
            elif eval_shards < cfg.minimum_eval_shards:
                summary["primary_status"] = "exploratory_insufficient_eval_shards"
            elif not provenance["preparation_manifest_matches_input"]:
                summary["primary_status"] = "exploratory_unverified_provenance"
            else:
                summary["primary_status"] = "eligible_for_limited_empirical_comparison"
            _write_json(output / "primary_selection.json", selection)
            primary_eval = parts["eval"]
        for split in ("dev", "eval"):
            baseline = -parts[split]["cv_min_clearance"].to_numpy(float)
            rows.append({"threshold_m": threshold, "ablation": "full", "model": "constant_velocity",
                         "seed": 42, "split": split, **metric_row(y[split], baseline, False)})
            if threshold == 2 and split == "eval":
                primary_scores["constant_velocity"] = baseline
        baseline_dev = -parts["dev"]["cv_min_clearance"].to_numpy(float)
        baseline_eval = -parts["eval"]["cv_min_clearance"].to_numpy(float)
        for budget in BUDGETS:
            policy = budget_policy(baseline_dev, budget)
            policy_rows.append({"threshold_m": threshold, "ablation": "full", "model": "constant_velocity",
                "seed": 42, "nominal_dev_budget": budget, **policy,
                **selection_metrics(y["eval"], apply_policy(baseline_eval, policy))})
        predictions = parts["eval"][["scenario_id", "source_file", label]].copy()
        predictions["constant_velocity"] = baseline_eval
        for ablation in ABLATIONS:
            columns = features[ablation]
            for name in MODEL_NAMES:
                for seed in cfg.seeds:
                    key = (ablation, name, seed)
                    if key in trained:
                        model, dev_score = trained[key]
                    else:
                        model = make_model(name, seed)
                        model.fit(parts["train"][columns], y["train"])
                        dev_score = model.predict_proba(parts["dev"][columns])[:, 1]
                    eval_score = model.predict_proba(parts["eval"][columns])[:, 1]
                    for split, score in (("dev", dev_score), ("eval", eval_score)):
                        rows.append({"threshold_m": threshold, "ablation": ablation, "model": name,
                                     "seed": seed, "split": split, **metric_row(y[split], score, True)})
                    predictions[f"{ablation}__{name}__{seed}"] = eval_score
                    if threshold == 2 and ablation == "full" and seed == 42:
                        primary_scores[name] = eval_score
                    for budget in BUDGETS:
                        policy = budget_policy(dev_score, budget)
                        policy_rows.append({"threshold_m": threshold, "ablation": ablation,
                            "model": name, "seed": seed, "nominal_dev_budget": budget, **policy,
                            **selection_metrics(y["eval"], apply_policy(eval_score, policy))})
        predictions.to_csv(private / f"predictions_{threshold}m.csv", index=False)
        summary["cohorts"][f"{threshold}m"]["status"] = "complete" if enough else "exploratory_complete"
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(output / "dev_fixed_policies.csv", index=False)
    if primary_eval is not None and summary["primary_selection"]:
        chosen = summary["primary_selection"]["model"]
        y_eval = primary_eval["label_2m"].to_numpy(int)
        if 0 < y_eval.sum() < len(y_eval):
            a, b = primary_scores[chosen], primary_scores["constant_velocity"]
            am, bm = metric_row(y_eval, a, True), metric_row(y_eval, b, False)
            summary["primary_comparison"] = {"ap_difference": am["ap"] - bm["ap"],
                "recall_at_10pct_difference": am["recall_budget_10pct"] - bm["recall_budget_10pct"]}
            summary["paired_bootstrap"] = paired_bootstrap(y_eval, a, b,
                primary_eval["source_file"].to_numpy(str), cfg)
            if summary["primary_status"] == "eligible_for_limited_empirical_comparison" and summary["paired_bootstrap"]["valid_draws"] < max(100, math.ceil(.8 * cfg.bootstrap_draws)):
                summary["primary_status"] = "exploratory_insufficient_bootstrap_draws"
            _plot_curves(output / "primary_curves.png", y_eval, primary_scores)
        else:
            summary["primary_status"] = "evaluation_missing_class_no_discrimination_conclusion"
        repeated = metrics[(metrics.threshold_m == 2) & (metrics.split == "eval") &
                           (metrics.ablation == "full") & (metrics.model == chosen)]
        summary["seed_stability"] = {"seeds": list(cfg.seeds), "n_seeds": len(repeated),
            "ap_min": repeated.ap.min(), "ap_max": repeated.ap.max(),
            "recall_10pct_min": repeated.recall_budget_10pct.min(),
            "recall_10pct_max": repeated.recall_budget_10pct.max()}
    if sha256_file(input_csv) != digest:
        raise RuntimeError("input_changed_during_experiment")
    _write_json(output / "summary.json", summary)
    (output / "report.md").write_text(_report(summary, metrics), encoding="utf-8")
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                    empirical_status=summary["primary_status"])
    _write_json(output / "run_manifest.json", manifest)
    return _jsonable(summary)
