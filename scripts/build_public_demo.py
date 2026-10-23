"""Build a small, explicit static research-publication bundle for GitHub Pages.

Only synthetic saved replays and publication notices are copied. The site does
not run inference, expose a Python API, include real records, or ship weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = {
    'demo/index.html': 'index.html',
    'demo/sidewalk_gallery.js': 'sidewalk_gallery.js',
    'licenses/WAYMO_DATASET_LICENSE_NON_COMMERCIAL_MARCH_2025.txt':
        'licenses/WAYMO_DATASET_LICENSE_NON_COMMERCIAL_MARCH_2025.txt',
    'models/NOTICE.txt': 'models/NOTICE.txt',
}


def build(destination: Path) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('public_demo_output_must_be_empty')
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for source_name, public_name in PUBLIC_FILES.items():
        source = ROOT / source_name
        if not source.is_file() or source.is_symlink():
            raise ValueError('required_publication_file_missing_or_symlink')
        target = destination / public_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append({'path': public_name, 'bytes': target.stat().st_size,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    (destination / '.nojekyll').write_text('', encoding='utf-8')
    manifest = {'purpose': 'non-commercial static research illustration',
                'mode': 'saved_synthetic_replays', 'model_inference_hosted': False,
                'raw_dataset_included': False, 'model_weights_included': False,
                'repository': 'https://github.com/thisisyk/sidewalk-delivery-robot-research',
                'full_agreement': 'licenses/WAYMO_DATASET_LICENSE_NON_COMMERCIAL_MARCH_2025.txt',
                'notice': 'models/NOTICE.txt', 'files': records}
    (destination / 'site-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.output)
    print(json.dumps({'mode': manifest['mode'], 'files': len(manifest['files']) + 2,
                      'model_inference_hosted': False, 'raw_dataset_included': False}))


if __name__ == '__main__':
    main()
