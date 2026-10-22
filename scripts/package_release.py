"""Create a local review ZIP from an explicit allowlist, never upload it."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=root/"results"/"real")
    p.add_argument("--output",type=Path,default=root.parent/"autonomous-delivery-risk-review.zip")
    args=p.parse_args()
    summary=args.results/"summary.json"
    report=args.results/"report.md"
    if not summary.is_file() or not report.is_file():
        p.error("Real-data evaluation has not completed; no release package will be created")
    result=json.loads(summary.read_text(encoding="utf-8"))
    if result.get("primary_status")!="eligible_for_limited_empirical_comparison":
        p.error("Primary analysis remains exploratory/insufficient; review evidence before packaging")
    review=root/"docs"/"case_review.md"
    if not review.is_file():
        p.error("Representative label/case review has not been recorded")
    for name in ("numeric_audit.json", "label_audit.json", "source_audit.json"):
        audit_path=args.results/name
        if not audit_path.is_file():
            p.error(f"Missing independent aggregate audit: {name}")
        audit=json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") not in ("pass", "pass_with_stated_limits"):
            p.error(f"Independent audit did not pass: {name}")
    files=[root/name for name in ("README.md","requirements.txt","requirements-preprocess.txt","pyproject.toml",".gitignore")]
    for directory in ("src","scripts","tests","configs","docs",".github"):
        files.extend(x for x in (root/directory).rglob("*") if x.is_file() and "__pycache__" not in x.parts and x.suffix in (".py", ".ps1", ".md", ".json", ".yml", ".yaml"))
    public_extras = {
        "demo": ("index.html", "example_run.js", "threshold_050_run.json", "learned_simulation.gif", "learned_simulation.png", "browser_preview.png", "scene_gallery.js", "urban_atlas.png", "urban_scenarios.gif", "urban_scenarios.png", "urban_scenarios.manifest.json", "urban_browser_preview.png", "sidewalk_gallery.js", "sidewalk_atlas.png", "sidewalk_scenarios.gif", "sidewalk_scenarios.png", "sidewalk_scenarios.manifest.json", "sidewalk_browser_preview.png", "assistance_browser_preview.png"),
        "results/extension": ("capacity_curve.csv", "extension_diagnostics.png", "extension_manifest.json", "extension_qa.json", "fixed_dev_threshold_transfer.csv", "paired_capacity_bootstrap.json", "score_reliability.csv", "subgroups_fixed_dev10.csv"),
        "results/queue_simulation": ("metrics.csv", "summary.json", "figure.png", "manifest.json"),
        "results/simulation": ("model_parity.json", "simulation_audit.json"),
        "results/urban": ("scenario_metrics.csv", "summary.json", "manifest.json", "visibility_audit.json", "release_audit.json"),
        "results/sidewalk": ("scenario_metrics.csv", "summary.json", "manifest.json", "geometry_motion_audit.json", "release_audit.json", "fallback_comparison.json", "fallback_comparison.csv", "assistance_audit.json"),
        "results/sidewalk_baseline": ("scenario_metrics.csv", "summary.json", "manifest.json", "geometry_motion_audit.json", "release_audit.json"),
    }
    if (root/"demo/index.html").is_file():
        parity = root/"results/simulation/model_parity.json"
        if not parity.is_file() or json.loads(parity.read_text(encoding="utf-8")).get("passed") is not True:
            p.error("Learned-model demo reproduction audit must pass before packaging")
        for name in ("geometry_motion_audit.json", "assistance_audit.json", "release_audit.json"):
            path = root/"results/sidewalk"/name
            if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("status") != "pass":
                p.error(f"Current sidewalk demo audit must pass before packaging: {name}")
    for directory, names in public_extras.items():
        files.extend(root/directory/name for name in names if (root/directory/name).is_file())
    allowed_results=("report.md","metrics.csv","dev_fixed_policies.csv","summary.json","primary_curves.png","run_manifest.json","primary_selection.json","feature_columns.json",
                     "numeric_audit.json","label_audit.json","source_audit.json","source_manifest.json","auth_cleanup.json","review_status.json")
    records=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(set(files)):
            if file.is_symlink() or root.resolve() not in file.resolve().parents:
                p.error("Review package cannot contain symlinks or files outside the project")
            data=file.read_bytes(); name=str(file.relative_to(root)).replace("\\","/")
            z.writestr(name,data)
            records.append({"path":name,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)})
        for name in allowed_results:
            file=args.results/name
            if file.is_file():
                data=file.read_bytes(); rel="results/real/"+name
                z.writestr(rel,data)
                records.append({"path":rel,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)})
        z.writestr("release_manifest.json",json.dumps({"publication_approval":"pending_user_review","files":records},indent=2))
    print(f"Local review package only (not uploaded): {args.output.resolve()}")


if __name__=="__main__": main()
