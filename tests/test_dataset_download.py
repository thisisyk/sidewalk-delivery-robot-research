"""Small synthetic files verify checksums and official-source boundaries only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dataset_download', ROOT / 'scripts/download_womd.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class DatasetDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b'synthetic checksum fixture; no Waymo records'
        self.row = {'path': 'training/training.tfrecord-00000-of-01000',
                    'uri': download.BUCKET + 'training/training.tfrecord-00000-of-01000',
                    'bytes': len(self.payload), 'sha256': hashlib.sha256(self.payload).hexdigest()}
        self.manifest = {'files': [self.row], 'file_count': 1, 'total_bytes': len(self.payload)}

    def test_missing_and_same_length_corruption_fail_checksum(self):
        path = self.root / self.row['path']
        self.assertFalse(download.verify_files(self.manifest, self.root)['passed'])
        path.parent.mkdir()
        path.write_bytes(self.payload)
        self.assertTrue(download.verify_files(self.manifest, self.root)['passed'])
        path.write_bytes(b'x' * len(self.payload))
        self.assertFalse(download.verify_files(self.manifest, self.root)['passed'])

    def test_registration_required_before_any_download(self):
        with patch.object(download.subprocess, 'run') as call:
            with self.assertRaisesRegex(ValueError, 'register_and_accept'):
                download.download_files(self.manifest, self.root, registered=False, gcloud='mock-gcloud')
            call.assert_not_called()

    def test_existing_corrupt_data_is_never_overwritten(self):
        path = self.root / self.row['path']
        path.parent.mkdir()
        path.write_bytes(b'keep this existing file')
        with patch.object(download.subprocess, 'run') as call:
            with self.assertRaisesRegex(ValueError, 'not_overwritten'):
                download.download_files(self.manifest, self.root, registered=True, gcloud='mock-gcloud')
            call.assert_not_called()
        self.assertEqual(path.read_bytes(), b'keep this existing file')

    def test_manifest_rejects_paths_and_nonofficial_sources(self):
        source = self.root / 'manifest.json'
        for name, uri in (('../escape', self.row['uri']), (self.row['path'], 'https://example.org/raw')):
            row = {**self.row, 'path': name, 'uri': uri}
            source.write_text(json.dumps({**self.manifest, 'files': [row]}), encoding='utf-8')
            with self.assertRaises(ValueError):
                download.load_manifest(source)

    def test_auth_download_verifies_before_promoting_partial(self):
        def fake_copy(arguments, **kwargs):
            Path(arguments[-1]).write_bytes(self.payload)
        with patch.object(download.subprocess, 'run', side_effect=fake_copy) as call:
            result = download.download_files(self.manifest, self.root, registered=True, gcloud='mock-gcloud')
        self.assertEqual(result['downloaded'], 1)
        self.assertEqual(call.call_args.args[0][3], self.row['uri'])
        self.assertEqual((self.root / self.row['path']).read_bytes(), self.payload)
