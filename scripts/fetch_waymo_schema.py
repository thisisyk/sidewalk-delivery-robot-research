"""Prepare official Apache-2.0 Waymo protobuf Python modules.

Compile pinned public .proto code, or extract pre-generated pure Python files
from an independently downloaded, hash-pinned official PyPI wheel. This script
never downloads the Waymo dataset, signs in, or accepts data-access terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import Request, urlopen
import zipfile


PINNED_COMMIT = "99a4cb3ff07e2fe06c2ce73da001f850f628e45a"
REPOSITORY = "https://github.com/waymo-research/waymo-open-dataset"
OFFICIAL_WHEEL = "waymo_open_dataset_tf_2_12_0-1.6.7-py3-none-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
OFFICIAL_WHEEL_SHA256 = "963462582955fe7c13c559c8fb712c5a7d8f2404486af63b7964b3b8a84872f8"
OFFICIAL_WHEEL_METADATA = "https://pypi.org/pypi/waymo-open-dataset-tf-2-12-0/1.6.7/json"
# Git blob hashes bind files to the inspected official commit. SHA-256 hashes
# are recorded in the generated manifest for ordinary provenance checks.
PROTO_FILES = {
    "waymo_open_dataset/protos/scenario.proto": "22b1c3c55cad11fe6601261eea603dc7bcdf0ec2",
    "waymo_open_dataset/protos/map.proto": "0e93fb3b9cda683dc2ce427fbe82bd4a9921bf1d",
    "waymo_open_dataset/protos/compressed_lidar.proto": "0b4c1c2f72e9d7a42306705d2a0a1735232666bb",
    "waymo_open_dataset/protos/camera_tokens.proto": "2d513532af2c74beab7abf9426d22067e1c6d9e9",
    "waymo_open_dataset/dataset.proto": "78d767903c21342447e0e74c6c6fd97075afce66",
    "waymo_open_dataset/label.proto": "908f7a1c58140ea0881615672a78bf143220450a",
    "waymo_open_dataset/protos/keypoint.proto": "b335981644f51adada43c1c90f65db90ee20f4ce",
    "waymo_open_dataset/protos/vector.proto": "0e70e95843c0d12ba4dce4dd438a003004fa1a54",
}


def git_blob_sha(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def extract_official_schema(wheel: Path, output_dir: Path) -> None:
    """Copy exactly eight verified Python modules; never extract native code."""
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != OFFICIAL_WHEEL_SHA256:
        raise RuntimeError("Wheel SHA256 does not match the official pinned PyPI release")
    manifest = {
        "repository": REPOSITORY,
        "schema_origin": "official_pypi_generated_modules",
        "package": "waymo-open-dataset-tf-2-12-0",
        "package_version": "1.6.7",
        "wheel_filename": OFFICIAL_WHEEL,
        "wheel_sha256": digest,
        "wheel_metadata": OFFICIAL_WHEEL_METADATA,
        "code_license": "Apache-2.0",
        "files": [],
    }
    # A whitelist prevents archive path traversal and excludes all native .so
    # files and package initializers that could import TensorFlow.
    with zipfile.ZipFile(wheel) as archive:
        for relative in PROTO_FILES:
            generated = relative.removesuffix(".proto") + "_pb2.py"
            content = archive.read(generated)
            if len(content) > 1024 * 1024:
                raise RuntimeError(f"Unexpectedly large generated module: {generated}")
            destination = output_dir / generated
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            manifest["files"].append({"path": generated, "sha256": hashlib.sha256(content).hexdigest()})
    for relative in ("waymo_open_dataset", "waymo_open_dataset/protos"):
        # The official wheel's top-level initializer is not needed for schema
        # loading; empty package files make this extraction self-contained.
        (output_dir / relative / "__init__.py").write_text("", encoding="utf-8")
    (output_dir / "schema_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "schema_extracted", "schema_root": str(output_dir), "wheel_sha256": digest, "module_count": len(PROTO_FILES)}, indent=2))


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=project / "vendor" / "waymo_proto")
    parser.add_argument("--output-dir", type=Path, default=project / "vendor" / "waymo_schema")
    parser.add_argument("--download-schema", action="store_true", help="Explicitly permit the 8 small official .proto source downloads")
    parser.add_argument("--wheel", type=Path, help="Extract Python schema only from the hash-pinned official 1.6.7 PyPI wheel; no protoc needed")
    args = parser.parse_args()
    source_dir, output_dir = args.source_dir.resolve(), args.output_dir.resolve()
    if args.wheel is not None:
        if args.download_schema:
            parser.error("--wheel and --download-schema are separate schema preparation methods")
        extract_official_schema(args.wheel.resolve(), output_dir)
        return
    manifest = {"repository": REPOSITORY, "commit": PINNED_COMMIT, "code_license": "Apache-2.0", "files": []}
    for relative, expected_blob in PROTO_FILES.items():
        source = source_dir / relative
        url = f"https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/{PINNED_COMMIT}/src/{relative}"
        if not source.is_file():
            if not args.download_schema:
                parser.error(f"Missing {source}. Pass --download-schema to fetch public schema code, or supply verified sources.")
            request = Request(url, headers={"User-Agent": "autonomous-delivery-risk-research/1.0"})
            with urlopen(request, timeout=30) as response:
                content = response.read(1024 * 1024 + 1)
            if len(content) > 1024 * 1024:
                raise RuntimeError(f"Unexpectedly large schema: {relative}")
            if git_blob_sha(content) != expected_blob:
                raise RuntimeError(f"Official schema content hash mismatch: {relative}")
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(content)
        content = source.read_bytes()
        if git_blob_sha(content) != expected_blob:
            raise RuntimeError(f"Local schema does not match pinned official source: {relative}")
        for imported in re.findall(r'^import\s+"([^"]+)";', content.decode("utf-8"), re.MULTILINE):
            if imported not in PROTO_FILES:
                raise RuntimeError(f"Unpinned dependency {imported} in {relative}")
        manifest["files"].append({"path": relative, "url": url, "git_blob_sha1": expected_blob, "sha256": hashlib.sha256(content).hexdigest()})
    try:
        import grpc_tools  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install grpcio-tools in this Python environment before compiling the schema.") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "grpc_tools.protoc", f"-I{source_dir}", f"--python_out={output_dir}"]
    command += [str(source_dir / relative) for relative in PROTO_FILES]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Schema compiler failed. If native compiler loading is unavailable, "
            "use --wheel with the official hash-pinned PyPI wheel to extract only "
            "the pre-generated Python protobuf modules. No native code is needed."
        ) from exc
    for relative in ("waymo_open_dataset", "waymo_open_dataset/protos"):
        (output_dir / relative / "__init__.py").touch()
    (source_dir / "schema_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "schema_compiled", "source_dir": str(source_dir), "schema_root": str(output_dir), "commit": PINNED_COMMIT}, indent=2))


if __name__ == "__main__":
    main()
