"""Export fixed training-only weights, audited against private held-out scores.

Requires licensed local samples and private predictions. Only weights, necessary
inference metadata, and aggregate parity measurements are written. No training
records, scenario IDs, raw coordinates, or per-scenario predictions are exported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delivery_risk.demo_model import load_frozen_demo_model, verify_parity
from delivery_risk.model_artifact import load_model_artifact, save_model_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--private-predictions", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "real")
    parser.add_argument("--output", type=Path, required=True,
                        help="Joblib path; sibling .manifest.json and model_parity.json are written.")
    parser.add_argument("--supporting-file", type=Path, action="append", default=[],
                        help="Include a relative path and SHA256 for a license or notice file.")
    args = parser.parse_args()
    parity_target = args.output.parent / "model_parity.json"
    if parity_target.exists():
        raise FileExistsError("model_parity_output_already_exists")
    model = load_frozen_demo_model(args.samples, results_dir=args.results_dir)
    audit = verify_parity(model, args.samples, args.results_dir / "metrics.csv",
                          args.private_predictions)
    manifest = save_model_artifact(model, args.output, parity_audit=audit,
                                   supporting_files=args.supporting_file)
    loaded = load_model_artifact(args.output, trusted=True,
                                expected_sha256=manifest["artifact"]["sha256"])
    frame = pd.read_csv(args.samples, dtype={"scenario_id": str})
    evaluation = frame.loc[(frame.split == "eval") & (frame.eligible_2m.astype(int) == 1)]
    features = evaluation[list(model.columns)]
    difference = float(np.max(np.abs(model.estimator.predict_proba(features) -
                                    loaded.estimator.predict_proba(features))))
    if difference > 1e-12:
        raise ValueError("serialized_model_failed_prediction_roundtrip")
    restored_audit = verify_parity(loaded, args.samples, args.results_dir / "metrics.csv",
                                   args.private_predictions)
    output_audit = {**restored_audit, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "artifact_exported": True,
                    "artifact_sha256": manifest["artifact"]["sha256"],
                    "roundtrip_prediction_max_absolute_difference": difference,
                    "raw_records_exported": False,
                    "per_scenario_predictions_exported": False,
                    "provenance": manifest["training_provenance"]}
    parity_target.write_text(json.dumps(output_audit, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": output_audit["passed"],
                      "artifact_sha256": output_audit["artifact_sha256"],
                      "artifact_bytes": manifest["artifact"]["bytes"],
                      "evaluation_rows": output_audit["evaluation_rows"],
                      "roundtrip_prediction_max_absolute_difference": difference,
                      "original_prediction_max_absolute_difference":
                          output_audit["original_prediction_max_absolute_difference"]}))


if __name__ == "__main__":
    main()
