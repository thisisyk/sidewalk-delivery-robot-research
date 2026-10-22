"""Render deterministic local-only trajectory cases after the frozen evaluation.

Case images are not evidence that road/video context was manually reviewed.
No images/coordinates or per-scene predictions are included in release ZIPs.
"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from delivery_risk.geometry import corners
from delivery_risk.samples import Protocol,history_view
from delivery_risk.waymo_reader import iter_scenarios


def plot_case(scene,out,label,score,kind):
    k=scene["current_time_index"]; cfg=Protocol()
    _,neighbors=history_view(scene,cfg)
    ids=[scene["sdc_track_index"],*[i for i,_ in neighbors]]
    ego=scene["tracks"][ids[0]]["states"][k]
    origin=np.array([ego["x"],ego["y"]])
    fig,ax=plt.subplots(figsize=(9,8))
    for m in scene.get("map_features",[]):
        points=m.get("polyline") or m.get("polygon")
        if points:
            xy=np.asarray(points)[:,:2]-origin
            if np.linalg.norm(xy,axis=1).min()<55:
                ax.plot(xy[:,0],xy[:,1],color="lightgray",lw=.5,zorder=0)
    for j,i in enumerate(ids):
        track=scene["tracks"][i]
        states=track["states"]; color="tab:blue" if j==0 else "tab:orange"
        past=np.array([[s["x"],s["y"]] for s in states[k-10:k+1] if s["valid"]])-origin
        future=np.array([[s["x"],s["y"]] for s in states[k:k+31] if s["valid"]])-origin
        ax.plot(past[:,0],past[:,1],color=color,lw=2,label="reference history" if j==0 else "neighbor history" if j==1 else None)
        ax.plot(future[:,0],future[:,1],color=color,lw=1.3,ls="--",label="reference future (label only)" if j==0 else "neighbor future (label only)" if j==1 else None)
        ax.add_patch(Polygon(corners(states[k])-origin,fill=False,edgecolor=color,lw=1.2))
    ax.set_aspect("equal"); ax.set_xlim(-55,55); ax.set_ylim(-55,55)
    ax.set(xlabel="Relative world X (m)",ylabel="Relative world Y (m)",title=f"{kind}: 2m proxy label={label}, model score={score:.3f}")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--schema-root",type=Path)
    args=p.parse_args()
    selection=json.loads((args.results/"primary_selection.json").read_text())
    model=selection["model"]; key=f"full__{model}__42"
    private=args.results.parent/"private"/args.results.name
    predictions=pd.read_csv(private/"predictions_2m.csv")
    policies=pd.read_csv(args.results/"dev_fixed_policies.csv")
    policy=policies[(policies.threshold_m==2)&(policies.model==model)&(policies.ablation=="full")&(policies.seed==42)&np.isclose(policies.nominal_dev_budget,.1)].iloc[0]
    cutoff=policy["cutoff"]
    # Cases strictly above/below cutoff; tied boundary belongs to randomized group.
    groups={"high_score_positive":predictions[(predictions.label_2m==1)&(predictions[key]>cutoff)].sort_values([key,"scenario_id"],ascending=[False,True]),"high_score_negative":predictions[(predictions.label_2m==0)&(predictions[key]>cutoff)].sort_values([key,"scenario_id"],ascending=[False,True]),"missed_positive":predictions[(predictions.label_2m==1)&(predictions[key]<cutoff)].sort_values([key,"scenario_id"])}
    chosen={kind:group.iloc[0] for kind,group in groups.items() if len(group)}
    out=private/"cases"; out.mkdir(exist_ok=True)
    checks=[]
    for filename in sorted({r.source_file for r in chosen.values()}):
        source=args.data_root/"validation"/filename
        for scene in iter_scenarios(source,schema_root=args.schema_root):
            for kind,row in chosen.items():
                if scene["scenario_id"]==row.scenario_id:
                    plot_case(scene,out/f"{kind}.png",int(row.label_2m),float(row[key]),kind)
                    checks.append({"kind":kind,"scenario_id":row.scenario_id,"source_file":filename,"label":int(row.label_2m),"score":float(row[key]),"human_reviewed":False})
    (out/"case_manifest.json").write_text(json.dumps(checks,indent=2))
    print(json.dumps({"local_cases":len(checks),"path":str(out.resolve()),"review_status":"visual_inspection_required"}))


if __name__=="__main__": main()
