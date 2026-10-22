"""Prepare local, officially obtained WOMD Scenario TFRecords, without network."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import platform

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pandas as pd
import numpy as np
from delivery_risk.samples import Protocol,extract_features,future_labels,history_view,_valid
from delivery_risk.waymo_reader import iter_scenarios


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def prepare_file(job):
    filename,split,cache,schema_root=job
    path=Path(filename)
    cfg=Protocol()
    sha=digest(path)
    project=Path(__file__).resolve().parents[1]
    code_files=[Path(__file__),*[project/"src"/"delivery_risk"/name for name in ("samples.py","geometry.py","waymo_reader.py")]]
    code_hash=hashlib.sha256(json.dumps({p.name:digest(p) for p in code_files},sort_keys=True).encode()).hexdigest()
    dest=Path(cache)/(path.name+".csv")
    meta=dest.with_suffix(".json")
    if dest.exists() and meta.exists():
        previous=json.loads(meta.read_text())
        if previous["source_sha256"]==sha and previous["protocol_sha256"]==cfg.fingerprint() and previous.get("extraction_code_sha256")==code_hash and previous["split"]==split and previous["csv_sha256"]==digest(dest):
            return previous
        raise ValueError(f"Cache changed: {dest}. Use new output directory.")
    counts=Counter()
    rows=[]
    started=time.time()
    for scene in iter_scenarios(path,schema_root=schema_root):
        counts["raw"]+=1
        try:
            _,near=history_view(scene,cfg)
            # Fast censoring audit before expensive constant-velocity geometry.
            k=scene["current_time_index"]
            for i in [scene["sdc_track_index"],*[idx for idx,_ in near]]:
                future=scene["tracks"][i]["states"][k+1:k+cfg.future_steps+1]
                if len(future)!=cfg.future_steps or not all(_valid(s) for s in future):
                    raise ValueError("future_censored")
            features,indices=extract_features(scene,cfg)
            labels=future_labels(scene,indices,features,cfg)
        except ValueError as exc:
            counts[str(exc)]+=1
            continue
        # Stable physical fingerprint used only for duplicate auditing, never input.
        ego=scene["tracks"][scene["sdc_track_index"]]["states"]
        physical=[[s["x"],s["y"],s["heading"]] for s in ego[k-10:k+31]]
        fingerprint=hashlib.sha256(json.dumps(physical).encode()).hexdigest()
        row={"scenario_id":scene["scenario_id"],"split":split,"source_file":path.name,"source_sha256":sha,"trajectory_sha256":fingerprint,**features,**labels}
        rows.append(row)
        counts["retained_complete"]+=1
        for d in cfg.thresholds_m:
            key=f"{d:g}m"
            if labels[f"eligible_{key}"]:
                counts[f"eligible_{key}"]+=1
                counts[f"positive_{key}"]+=labels[f"label_{key}"]
            else:
                counts[f"current_event_{key}"]+=1
        if counts["raw"]%50==0:
            print(f"{path.name}: {counts['raw']} read, {len(rows)} complete",flush=True)
    dest.parent.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(dest,index=False)
    metadata={"source_file":path.name,"source_sha256":sha,"source_bytes":path.stat().st_size,"split":split,"protocol_sha256":cfg.fingerprint(),"extraction_code_sha256":code_hash,"counts":dict(counts),"csv":str(dest),"csv_sha256":digest(dest),"elapsed_seconds":time.time()-started}
    meta.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(f"DONE {path.name}: {dict(counts)}",flush=True)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True,help="Contains training/ and validation/")
    p.add_argument("--output",type=Path,default=Path("artifacts/prepared"))
    p.add_argument("--schema-root",type=Path)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--train-shards",type=int,default=12)
    p.add_argument("--validation-shards",type=int,default=10)
    args=p.parse_args()
    if args.train_shards<1 or args.validation_shards<2 or args.workers<1:
        p.error("Need at least one train, two validation shards, and one worker")
    args.output.mkdir(parents=True,exist_ok=True)
    train=sorted((args.data_root/"training").glob("training.tfrecord-*"))[:args.train_shards]
    val=sorted((args.data_root/"validation").glob("validation.tfrecord-*"))[:args.validation_shards]
    if len(train)!=args.train_shards or len(val)!=args.validation_shards:
        p.error(f"Expected fixed {args.train_shards} train/{args.validation_shards} val shards; found {len(train)}/{len(val)}. Download complete set or explicitly revise protocol before evaluating.")
    # Split established from filenames before any outcome inspection.
    split_manifest={"train":[x.name for x in train],"dev":[x.name for i,x in enumerate(val) if i%2==0],"eval":[x.name for i,x in enumerate(val) if i%2==1]}
    frozen=args.output/"split_manifest.json"
    # Normalize tuples to the same representation used when JSON is read back;
    # otherwise an unchanged Protocol.thresholds_m tuple mismatches its list.
    manifest=json.loads(json.dumps({"dataset":"WOMD v1.3.1 Scenario","protocol":asdict(Protocol()),"protocol_sha256":Protocol().fingerprint(),"files":split_manifest}))
    if frozen.exists() and json.loads(frozen.read_text())!=manifest:
        raise ValueError("Existing split manifest differs; use a new preparation directory")
    frozen.write_text(json.dumps(manifest,indent=2)+"\n")
    jobs=[(str(path.resolve()),split,str((args.output/"shards").resolve()),str(args.schema_root.resolve()) if args.schema_root else None) for split in split_manifest for path in (train if split=="train" else val) if path.name in split_manifest[split]]
    if args.workers==1:
        meta=[prepare_file(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            meta=list(pool.map(prepare_file,jobs))
    frames=[]
    for m in meta:
        if m["counts"].get("retained_complete",0):
            frames.append(pd.read_csv(m["csv"]))
    if not frames:
        raise ValueError("No complete samples after censoring")
    frame=pd.concat(frames,ignore_index=True)
    if frame.scenario_id.duplicated().any():
        raise ValueError("Duplicate scenario IDs; investigate dataset selection")
    if frame.groupby("trajectory_sha256")["split"].nunique().gt(1).any():
        raise ValueError("Identical reference trajectories across splits")
    dest=args.output/"samples.csv"
    frame.to_csv(dest,index=False)
    provenance={"source_type":"official_womd_scenario","dataset":"WOMD v1.3.1","synthetic":False,"prepared_at":datetime.now(timezone.utc).isoformat(),"protocol":asdict(Protocol()),"protocol_sha256":Protocol().fingerprint(),"split_manifest_sha256":digest(frozen),"samples_sha256":digest(dest),"sources":meta,"software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},"no_real_intervention_labels":True,"raw_log_disjointness_verified":False}
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print(f"Prepared {len(frame)} scenes: {dest}",flush=True)


if __name__=="__main__":
    main()
