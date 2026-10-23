"""The public website must exclude the real dataset and model binary."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_public_demo', ROOT / 'scripts/build_public_demo.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PublicBundleTests(unittest.TestCase):
    def test_publication_contains_only_saved_replays_and_notices(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'site'
            manifest = builder.build(destination)
            expected = set(builder.PUBLIC_FILES.values()) | {'.nojekyll', 'site-manifest.json'}
            actual = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
            self.assertEqual(actual, expected)
            self.assertFalse(manifest['model_inference_hosted'])
            self.assertFalse(manifest['model_weights_included'])
            self.assertFalse(manifest['raw_dataset_included'])
            for source, name in builder.PUBLIC_FILES.items():
                self.assertEqual((ROOT / source).read_bytes(), (destination / name).read_bytes())
            self.assertEqual(json.loads((destination / 'site-manifest.json').read_text())['files'], manifest['files'])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            sentinel = destination / 'keep.txt'
            sentinel.write_text('existing work')
            with self.assertRaisesRegex(ValueError, 'must_be_empty'):
                builder.build(destination)
            self.assertEqual(sentinel.read_text(), 'existing work')
