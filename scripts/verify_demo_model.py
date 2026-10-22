"""Re-fit the frozen training-only model and audit its original score parity.

Requires the licensed local prepared samples. Writes aggregate JSON, no weights
or per-scene data. This is reproduction of the original model, not selection.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delivery_risk.demo_model import load_frozen_demo_model, verify_parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "real")
    parser.add_argument("--original-private-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = load_frozen_demo_model(args.samples, results_dir=args.results)
    audit = verify_parity(model, args.samples, args.results / "metrics.csv",
                          args.original_private_predictions)
    audit["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": audit["passed"], "eval_ap": audit["reproduced_eval_ap"],
                      "model": audit["model"], "features": audit["feature_count"],
                      "weights_written": False}))


if __name__ == "__main__":
    main()
