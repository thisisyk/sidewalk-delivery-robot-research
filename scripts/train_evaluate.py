"""Run the frozen real-data experiment from a prepared samples.csv file."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delivery_risk.experiment import ExperimentConfig, run_experiment
from delivery_risk.experiment import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Prepared real WOMD samples CSV")
    parser.add_argument("--output", type=Path, default=Path("results/real"))
    parser.add_argument("--bootstrap-draws", type=int, default=300)
    args = parser.parse_args()
    provenance_path = args.input.parent / "provenance.json"
    if not provenance_path.exists():
        parser.error("Missing provenance.json; run scripts/prepare.py on official Scenario TFRecords first")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("synthetic") is not False or provenance.get("source_type") != "official_womd_scenario":
        parser.error("This command accepts only prepared official WOMD data, not synthetic examples")
    if provenance.get("samples_sha256") != sha256_file(args.input):
        parser.error("Prepared CSV hash does not match provenance")
    cfg = ExperimentConfig(bootstrap_draws=args.bootstrap_draws)
    result = run_experiment(args.input, args.output, cfg)
    print(json.dumps({"status": result["primary_status"], "output": str(args.output.resolve()),
                      "approval": "pending_user_review"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
