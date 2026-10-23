"""Distribute a browser-local non-commercial research experiment.

The visitor downloads the frozen model and runs it locally in a worker. No
public inference server, real records, training rows or pickle are included.
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
    'demo/browser-compute.js': 'browser-compute.js',
    'demo/compute-worker.js': 'compute-worker.js',
    'demo/browser_bootstrap.py': 'browser_bootstrap.py',
    'demo/frozen_model.json': 'frozen_model.json',
    'licenses/WAYMO_DATASET_LICENSE_NON_COMMERCIAL_MARCH_2025.txt':
        'licenses/WAYMO_DATASET_LICENSE_NON_COMMERCIAL_MARCH_2025.txt',
    'models/NOTICE.txt': 'models/NOTICE.txt',
}
BROWSER_MODULES = ('__init__', 'browser_model', 'geometry', 'samples', 'simulation',
                   'urban_worlds', 'urban_simulation', 'sidewalk_worlds', 'assistance', 'sidewalk_simulation')
RUNTIME_VERSION = '314.0.7'


def build(destination: Path) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('public_demo_output_must_be_empty')
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    all_files = dict(PUBLIC_FILES)
    for name in BROWSER_MODULES:
        all_files[f'src/delivery_risk/{name}.py'] = f'python/delivery_risk/{name}.py'
    runtime_prefix = f'runtime/pyodide-{RUNTIME_VERSION}'
    runtime_root = ROOT / 'demo' / runtime_prefix
    provenance = json.loads((runtime_root / 'runtime-provenance.json').read_text(encoding='utf-8'))
    if provenance['version'] != RUNTIME_VERSION:
        raise ValueError('fixed_browser_runtime_required')
    runtime_paths = {row['path'] for row in provenance['files']}
    runtime_paths.update(row['path'] for row in provenance['embedded_notices_extracted'])
    runtime_paths.update(('LICENSE-PYODIDE.txt', 'NOTICE-RUNTIME.txt', 'runtime-provenance.json'))
    actual_paths = {path.relative_to(runtime_root).as_posix() for path in runtime_root.rglob('*') if path.is_file()}
    if actual_paths != runtime_paths:
        raise ValueError('runtime_directory_differs_from_explicit_provenance')
    for row in provenance['files'] + provenance['embedded_notices_extracted']:
        if hashlib.sha256((runtime_root / row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('browser_runtime_checksum_mismatch')
    for name in sorted(runtime_paths):
        all_files[f'demo/{runtime_prefix}/{name}'] = f'{runtime_prefix}/{name}'
    for source_name, public_name in all_files.items():
        source = ROOT / source_name
        if not source.is_file() or source.is_symlink():
            raise ValueError('required_publication_file_missing_or_symlink')
        target = destination / public_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append({'path': public_name, 'bytes': target.stat().st_size,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    (destination / '.nojekyll').write_text('', encoding='utf-8')
    by_path = {row['path']: row for row in records}
    resource = lambda name: {'url': name, 'sha256': by_path[name]['sha256']}
    compute = {'format': 'sidewalk-browser-compute-v1',
               'runtime': {'version': RUNTIME_VERSION, 'indexURL': runtime_prefix + '/'},
               'model': resource('frozen_model.json'), 'bootstrap': resource('browser_bootstrap.py'),
               'modules': [{'name': f'delivery_risk/{name}.py', **resource(f'python/delivery_risk/{name}.py')}
                           for name in BROWSER_MODULES],
               'agreement': 'licenses/WAYMO_DATASET_LICENSE_NON_COMMERCIAL_MARCH_2025.txt',
               'notice': 'models/NOTICE.txt', 'server_inference': False, 'personal_noncommercial_experiment_only': True}
    compute_target = destination / 'compute-manifest.json'
    compute_target.write_text(json.dumps(compute, indent=2) + '\n', encoding='utf-8')
    records.append({'path': 'compute-manifest.json', 'bytes': compute_target.stat().st_size,
                    'sha256': hashlib.sha256(compute_target.read_bytes()).hexdigest()})
    manifest = {'purpose': 'downloadable model and software for non-commercial personal research experiments',
                'mode': 'browser_local_model_compute', 'model_inference_hosted': False, 'browser_model_inference': True,
                'raw_dataset_included': False, 'model_weights_included': True, 'pickle_included': False,
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
                      'model_inference_hosted': False, 'raw_dataset_included': False, 'browser_model_inference': True}))


if __name__ == '__main__':
    main()
