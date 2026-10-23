"""Browser bridge for the unchanged research simulator and frozen JSON model.

This file imports NumPy and the exact source modules published with the demo.
It does not load scikit-learn, joblib, private samples, or saved trajectories.
Worker-supplied configuration is parsed as JSON, never executed as Python.
"""
from dataclasses import asdict, fields
import json
import platform

import numpy as np

from delivery_risk.assistance import ASSISTANCE_VERSION
from delivery_risk.browser_model import load_browser_model
from delivery_risk.sidewalk_worlds import SIDEWALK_CATALOG
from delivery_risk import sidewalk_simulation as _sidewalk


_browser_model = None
_POLICY_NAMES = {
    "learned_with_fallback": "模型 + 停滞兜底 + 协助指令",
    "learned": "原纯模型门控（无兜底）",
    "fallback_only": "仅加停滞求助（旧处理器）",
    "all": "等待2秒即请求",
    "no_assist": "仅自主等待",
}


def _report(phase, message, **details):
    """UI observations cannot change or interrupt simulated state transitions."""
    callback = globals().get("_browser_progress_callback")
    if callback is not None:
        try:
            callback(phase, message, json.dumps(details, ensure_ascii=False, allow_nan=False))
        except Exception:
            pass


def initialize_browser_model(model_path):
    """Return the local server's info schema with explicit browser provenance."""
    global _browser_model
    _browser_model = load_browser_model(model_path)
    info = {
        "model": "HistGradientBoosting · full · seed42",
        "training_rows": _browser_model.fit_audit["training_rows"],
        "feature_count": len(_browser_model.columns),
        "dev_threshold": _browser_model.dev_policy["cutoff"],
        "eval_ap": .7302762063131997,
        "baseline_ap": .44926474314513035,
        "synthetic": True,
        "domain": "robot",
        "scenario_catalog": SIDEWALK_CATALOG,
        "assistance_module_version": ASSISTANCE_VERSION,
        "primary_strategy": "learned_with_fallback_and_explicit_simulated_operator_commands",
        "browser_runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "inference": "exported_frozen_histgb_tree_values_and_original_feature_pipeline",
            "source_model_artifact_sha256": _browser_model.source_artifact_sha256,
            "raw_data_downloaded": False,
            "saved_trajectory_used_for_computation": False,
        },
        "notes": [
            "浏览器内运行同一特征提取、训练模型树和原 Python 仿真；每次按输入参数重新计算",
            "配送机器人与行人仅在人行道/斑马线通行；汽车仅机动车道",
            "实际冻结 HistGB 评分；原训练域为道路车辆，机器人属于域外迁移示意",
            "模型分数未经机器人域校准；运动、通行域与协助机制是显式规则",
        ],
    }
    return json.dumps(info, ensure_ascii=False, allow_nan=False)


def _run_with_progress(cfg, policy, *, capture, policy_index, total_policies):
    """Observe real simulator time without editing its original implementation.

    The wrapper forwards every sidewalk_actors call and result unchanged. It
    reports the time already requested by the simulator at five-second steps.
    Callback failures are ignored, and the function is always restored.
    """
    original_actors = _sidewalk.sidewalk_actors
    last_reported_bucket = -1

    def observed_actors(t, *args, **kwargs):
        nonlocal last_reported_bucket
        actors = original_actors(t, *args, **kwargs)
        bucket = int(float(t) // 5)
        if bucket > last_reported_bucket:
            last_reported_bucket = bucket
            _report("simulate", f"正在计算：{_POLICY_NAMES[policy]} · {float(t):.0f}/{cfg.duration_s:.0f} 模拟秒",
                    simulated_seconds=float(t), duration_s=cfg.duration_s,
                    policy=policy, policy_index=policy_index, total_policies=total_policies)
        return actors

    _sidewalk.sidewalk_actors = observed_actors
    try:
        return _sidewalk._run_sidewalk(_browser_model, cfg, policy, capture)
    finally:
        _sidewalk.sidewalk_actors = original_actors


def run_browser_simulation(configuration_json, compare=False):
    """Calculate a fresh main strategy and optional complete paired controls."""
    if _browser_model is None:
        raise RuntimeError("browser_model_not_initialized")
    requested = json.loads(configuration_json)
    allowed = {field.name for field in fields(_sidewalk.SidewalkConfig)}
    if not isinstance(requested, dict) or not set(requested).issubset(allowed):
        raise ValueError("unsupported_request_fields")
    cfg = _sidewalk.SidewalkConfig(**requested)
    cfg.validate()
    for name in ("fleet", "operators", "seed"):
        if type(getattr(cfg, name)) is not int:
            raise ValueError(f"integer_configuration_required:{name}")
    if type(compare) is not bool:
        raise ValueError("comparison_flag_must_be_boolean")
    policy = "learned_with_fallback" if cfg.fallback_enabled else "learned"
    comparisons = ("learned", "fallback_only", "all", "no_assist") if cfg.fallback_enabled else ("all", "no_assist")
    total = 1 + len(comparisons) if compare else 1
    _report("simulate", "正在按当前参数计算新轨迹、观测历史、模型评分和协助动作…",
            policy=policy, policy_index=1, total_policies=total,
            simulated_seconds=0, duration_s=cfg.duration_s, config=asdict(cfg))
    result = _run_with_progress(cfg, policy, capture=True, policy_index=1, total_policies=total)
    if compare:
        summaries = [result["summary"]]
        for index, comparison in enumerate(comparisons, 2):
            summaries.append(_run_with_progress(cfg, comparison, capture=False,
                                               policy_index=index, total_policies=total)["summary"])
        result["policies"] = [{"name": _POLICY_NAMES[row["policy"]], **row} for row in summaries]
    _report("serialize", "模型和仿真计算完成，正在准备动画与协助记录…",
            simulated_seconds=cfg.duration_s, duration_s=cfg.duration_s,
            policy_index=total, total_policies=total)
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
