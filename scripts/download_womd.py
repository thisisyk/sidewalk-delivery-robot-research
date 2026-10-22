"""Obtain the frozen shards from Waymo with each recipient's own authorized account.

This script does not sign in, accept terms, issue credentials, mirror data, or
bypass official access control. With no mode selected, it only lists sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUCKET = 'gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/'
PATH_PATTERN = re.compile(r'(training/training\.tfrecord-\d{5}-of-01000|validation/validation\.tfrecord-\d{5}-of-00150)')


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding='utf-8'))
    rows = manifest['files']
    seen = set()
    for row in rows:
        name = row['path']
        if not PATH_PATTERN.fullmatch(name) or name in seen:
            raise ValueError('invalid_or_duplicate_official_shard_path')
        seen.add(name)
        if row['uri'] != BUCKET + name:
            raise ValueError('official_bucket_uri_required')
        if not isinstance(row['bytes'], int) or isinstance(row['bytes'], bool) or row['bytes'] <= 0:
            raise ValueError('invalid_shard_size')
        if not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise ValueError('invalid_shard_sha256')
    if len(rows) != manifest['file_count'] or sum(r['bytes'] for r in rows) != manifest['total_bytes']:
        raise ValueError('manifest_totals_mismatch')
    return manifest


def matches(path: Path, row: dict) -> bool:
    if not path.is_file() or path.is_symlink() or path.stat().st_size != row['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == row['sha256']


def verify_files(manifest: dict, destination: Path) -> dict:
    failed = [row['path'] for row in manifest['files'] if not matches(destination / row['path'], row)]
    return {'mode': 'verify_only', 'expected_files': len(manifest['files']),
            'verified_files': len(manifest['files']) - len(failed), 'failed_files': failed,
            'expected_bytes': manifest['total_bytes'], 'passed': not failed,
            'raw_data_uploaded': False}


def download_files(manifest: dict, destination: Path, *, registered: bool, gcloud: str | None = None) -> dict:
    if not registered:
        raise ValueError('register_and_accept_waymo_terms_first_then_use_registered_with_waymo')
    executable = gcloud or shutil.which('gcloud') or shutil.which('gcloud.cmd')
    if not executable:
        raise ValueError('install_official_google_cloud_cli_and_authenticate_your_own_account_first')
    downloaded = skipped = 0
    for row in manifest['files']:
        path = destination / row['path']
        if path.exists():
            if not matches(path, row):
                raise ValueError(f'existing_file_fails_checksum_not_overwritten: {row["path"]}')
            skipped += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + '.partial')
        if partial.exists():
            raise ValueError(f'partial_file_exists_check_or_remove_manually: {row["path"]}')
        subprocess.run([executable, 'storage', 'cp', row['uri'], str(partial)], check=True)
        if not matches(partial, row):
            raise ValueError(f'download_failed_checksum_partial_retained: {row["path"]}')
        partial.rename(path)
        downloaded += 1
    return {'mode': 'official_authenticated_download', 'downloaded': downloaded, 'existing_verified': skipped,
            'verified_bytes': manifest['total_bytes'], 'raw_data_uploaded': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'configs/womd_download_manifest.json')
    parser.add_argument('--destination', type=Path, default=ROOT / 'data/womd')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--download', action='store_true')
    parser.add_argument('--registered-with-waymo', action='store_true', help='Affirm that you registered and accepted the official Waymo terms; access is still enforced by Waymo.')
    parser.add_argument('--gcloud', help='Optional path to the official gcloud executable')
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.verify_only:
        result = verify_files(manifest, args.destination)
    elif args.download:
        try:
            result = download_files(manifest, args.destination, registered=args.registered_with_waymo, gcloud=args.gcloud)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        result = {'mode': 'list_only', 'license': manifest['license_url'], 'file_count': manifest['file_count'],
                  'total_bytes': manifest['total_bytes'], 'official_sources': [r['uri'] for r in manifest['files']],
                  'authentication_or_download_performed': False}
    print(json.dumps(result, indent=2))
    if result.get('passed') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
