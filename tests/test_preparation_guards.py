"""Temporary software fixtures check preparation/CLI guards, not study outcomes."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(f"guard_test_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreparationGuardTests(unittest.TestCase):
    def setUp(self):
        self.module = script("prepare")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "training.tfrecord-unit-fixture"
        self.raw.write_bytes(b"unit fixture only; parser is mocked")
        self.job = (str(self.raw), "train", str(self.root / "cache"), None)

    def build_empty_cache(self):
        with patch.object(self.module, "iter_scenarios", return_value=iter(())) as reader:
            with redirect_stdout(io.StringIO()):
                result = self.module.prepare_file(self.job)
            self.assertEqual(reader.call_count, 1)
        return result

    def test_matching_cache_reuses_and_tampered_csv_rejects(self):
        first = self.build_empty_cache()
        with patch.object(self.module, "iter_scenarios", side_effect=AssertionError("must reuse cache")):
            self.assertEqual(self.module.prepare_file(self.job), first)
        Path(first["csv"]).write_text("tampered cached output", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cache changed"):
            self.module.prepare_file(self.job)

    def test_extraction_code_hash_change_rejects_cache(self):
        self.build_empty_cache()
        original = self.module.digest

        def changed_digest(path):
            return "f" * 64 if path.name == "samples.py" else original(path)

        with patch.object(self.module, "digest", side_effect=changed_digest):
            with self.assertRaisesRegex(ValueError, "Cache changed"):
                self.module.prepare_file(self.job)

    def test_unchanged_split_manifest_can_resume(self):
        data_root = self.root / "raw"
        for split, count in (("training", 1), ("validation", 2)):
            folder = data_root / split
            folder.mkdir(parents=True)
            for index in range(count):
                (folder / f"{split}.tfrecord-{index:05d}-of-{count:05d}").write_bytes(b"unit fixture")
        output = self.root / "prepared"

        def fake_prepare(job):
            filename, split, cache, _ = job
            filename = Path(filename)
            csv = Path(cache) / (filename.name + ".csv")
            csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"scenario_id": filename.name, "split": split,
                           "trajectory_sha256": hashlib.sha256(filename.name.encode()).hexdigest()}]).to_csv(csv, index=False)
            return {"counts": {"retained_complete": 1}, "csv": str(csv)}

        argv = ["prepare.py", "--data-root", str(data_root), "--output", str(output),
                "--workers", "1", "--train-shards", "1", "--validation-shards", "2"]
        with patch.object(self.module, "prepare_file", side_effect=fake_prepare):
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                self.module.main()
                first = (output / "split_manifest.json").read_bytes()
                self.module.main()
        self.assertEqual(first, (output / "split_manifest.json").read_bytes())


class TrainingCliGuardTests(unittest.TestCase):
    def test_untrusted_input_never_reaches_training(self):
        module = script("train_evaluate")
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "fixture.csv"
            source.write_text("not training data", encoding="utf-8")
            for case in (None, {"synthetic": True, "source_type": "official_womd_scenario"},
                         {"synthetic": False, "source_type": "official_womd_scenario", "samples_sha256": "0" * 64}):
                with self.subTest(provenance=case):
                    provenance = source.parent / "provenance.json"
                    if case is not None:
                        provenance.write_text(json.dumps(case), encoding="utf-8")
                    with patch.object(module, "run_experiment") as train:
                        with patch("sys.argv", ["train_evaluate.py", "--input", str(source)]):
                            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
                                module.main()
                    self.assertEqual(exit_info.exception.code, 2)
                    train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
