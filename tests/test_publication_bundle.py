"""Browser model distribution must exclude real records and executable pickle."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_public_demo', ROOT / 'scripts/build_public_demo.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PublicBundleTests(unittest.TestCase):
    def test_browser_publication_includes_verified_model_but_no_real_records_or_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'site'
            manifest = builder.build(destination)
            expected = {row['path'] for row in manifest['files']} | {'.nojekyll', 'site-manifest.json'}
            actual = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
            self.assertEqual(actual, expected)
            self.assertFalse(manifest['model_inference_hosted'])
            self.assertTrue(manifest['model_weights_included'])
            self.assertTrue(manifest['browser_model_inference'])
            self.assertFalse(manifest['pickle_included'])
            self.assertFalse(manifest['raw_dataset_included'])
            self.assertTrue((destination / 'frozen_model.json').is_file())
            self.assertTrue((destination / manifest['full_agreement']).is_file())
            runtime = destination / f'runtime/pyodide-{builder.RUNTIME_VERSION}'
            provenance = json.loads((runtime / 'runtime-provenance.json').read_text())
            python_license = next(row for row in provenance['files'] if row['path'] == 'LICENSE-CPYTHON.txt')
            self.assertEqual(python_license['source'],
                             'https://raw.githubusercontent.com/python/cpython/v3.14.2/LICENSE')
            license_bytes = (runtime / python_license['path']).read_bytes()
            self.assertEqual(hashlib.sha256(license_bytes).hexdigest(), python_license['sha256'])
            self.assertIn(b'PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2', license_bytes)
            self.assertFalse(any(p.suffix in {'.pkl', '.joblib', '.tfrecord', '.csv'} for p in destination.rglob('*')))
            compute = json.loads((destination / 'compute-manifest.json').read_text())
            self.assertFalse(compute['server_inference'])
            self.assertEqual(len(compute['modules']), 10)
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
