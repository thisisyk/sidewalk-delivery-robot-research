"""Past-only features, separately computed future labels and explicit censoring."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
from .geometry import clearance, extrapolate, clearance_many


@dataclass(frozen=True)
class Protocol:
    history_steps: int = 10
    future_steps: int = 30
    dt: float = 0.1
    radius_m: float = 50.0
    max_neighbors: int = 16
    thresholds_m: tuple = (1.0,2.0,3.0)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


def _valid(state):
    keys = ("x","y","length","width","heading","vx","vy")
    return bool(state.get("valid")) and all(k in state for k in keys) and np.isfinite([state[k] for k in keys]).all() and min(state["length"],state["width"]) > 0


def history_view(scene: dict, cfg: Protocol):
    """This is the only input boundary used by feature extraction."""
    k = scene["current_time_index"]
    if k < cfg.history_steps:
        raise ValueError("insufficient_history")
    ts = np.asarray(scene["timestamps"][k-cfg.history_steps:k+1],dtype=float)
    if len(ts)!=cfg.history_steps+1 or not np.allclose(np.diff(ts),cfg.dt,atol=0.002):
        raise ValueError("unexpected_timestamps")
    ego_track = scene["tracks"][scene["sdc_track_index"]]
    hist = ego_track["states"][k-cfg.history_steps:k+1]
    if len(hist)!=cfg.history_steps+1 or not all(_valid(s) for s in hist):
        raise ValueError("invalid_ego_history")
    ego = hist[-1]
    neighbors=[]
    for i,track in enumerate(scene["tracks"]):
        if i==scene["sdc_track_index"] or len(track["states"])<=k:
            continue
        s=track["states"][k]
        if _valid(s) and track["type"] in (1,2,3):
            dist=float(np.hypot(s["x"]-ego["x"],s["y"]-ego["y"]))
            if dist<=cfg.radius_m:
                neighbors.append((dist,track["id"],i,track))
    neighbors.sort(key=lambda r:(r[0],r[1]))
    chosen=neighbors[:cfg.max_neighbors]
    if not chosen:
        raise ValueError("no_current_neighbors")
    return hist,[(i,{"id":t["id"],"type":t["type"],"states":t["states"][k-cfg.history_steps:k+1]}) for _,_,i,t in chosen]


def extract_features(scene: dict,cfg: Protocol):
    hist, neighbors=history_view(scene,cfg)
    ego=hist[-1]
    speed=lambda s:float(np.hypot(s["vx"],s["vy"]))
    speeds=np.array([speed(s) for s in hist])
    heading=np.unwrap([s["heading"] for s in hist])
    features={"ego_speed":speeds[-1],"ego_accel":(speeds[-1]-speeds[0])/(cfg.history_steps*cfg.dt),"ego_speed_std":speeds.std(),"ego_yaw_change":heading[-1]-heading[0]}
    c,s=np.cos(ego["heading"]),np.sin(ego["heading"])
    rotation=np.array([[c,s],[-s,c]])
    features.update({"neighbor_count":len(neighbors),"pedestrian_count":sum(n["type"]==2 for _,n in neighbors),"cyclist_count":sum(n["type"]==3 for _,n in neighbors)})
    current_min,cv_min,ttc=100.,100.,cfg.future_steps*cfg.dt+1
    for j in range(cfg.max_neighbors):
        vals=[0.]*11
        if j<len(neighbors):
            _,n=neighbors[j]
            state=n["states"][-1]
            rel=rotation@np.array([state["x"]-ego["x"],state["y"]-ego["y"]])
            vel=rotation@np.array([state["vx"]-ego["vx"],state["vy"]-ego["vy"]])
            d=clearance(ego,state)
            current_min=min(current_min,d)
            prev=n["states"][0]
            prev_ok=_valid(prev)
            closing=0. if not prev_ok else (np.hypot(prev["x"]-hist[0]["x"],prev["y"]-hist[0]["y"])-np.linalg.norm(rel))/(cfg.history_steps*cfg.dt)
            relative_heading=(state["heading"]-ego["heading"]+np.pi)%(2*np.pi)-np.pi
            vals=[1.,float(n["type"]),*rel,*vel,d,speed(state),float(prev_ok),closing,relative_heading]
        for key,val in zip(("present","type","x","y","vx","vy","clearance","speed","history_valid","closing","heading"),vals):
            features[f"n{j}_{key}"]=float(val)
    cv_ego=[]; cv_other=[]
    for _,neighbor in neighbors:
        for step in range(1,cfg.future_steps+1):
            cv_ego.append(extrapolate(ego,step*cfg.dt))
            cv_other.append(extrapolate(neighbor["states"][-1],step*cfg.dt))
    cv_distances=clearance_many(cv_ego,cv_other).reshape(len(neighbors),cfg.future_steps)
    cv_min=float(cv_distances.min())
    hits=np.flatnonzero((cv_distances<=2.).any(axis=0))
    ttc=float((hits[0]+1)*cfg.dt) if len(hits) else cfg.future_steps*cfg.dt+1
    features.update({"current_min_clearance":current_min,"cv_min_clearance":cv_min,"cv_time_to_2m":ttc})
    # Static geometry is allowed, but never future traffic signals / sdc_paths.
    for kind in ("crosswalk","stop_sign","road_edge","lane"):
        distances=[]
        for m in scene.get("map_features",[]):
            if m["type"]==kind:
                points=m.get("polygon") or m.get("polyline") or ([m["position"]] if "position" in m else [])
                if points:
                    arr=np.asarray(points)
                    distances.append(float(np.linalg.norm(arr[:,:2]-[ego["x"],ego["y"]],axis=1).min()))
        features[f"map_{kind}_vertex_distance"]=min(distances,default=100.)
        features[f"map_{kind}_within30"]=float(sum(d<=30 for d in distances))
    return features,[i for i,_ in neighbors]


def future_labels(scene: dict,indices: list[int],features: dict,cfg: Protocol):
    k=scene["current_time_index"]
    stop=k+cfg.future_steps+1
    if stop>len(scene["timestamps"]) or not np.allclose(np.diff(scene["timestamps"][k:stop]),cfg.dt,atol=0.002):
        raise ValueError("future_timestamp_coverage")
    ids=[scene["sdc_track_index"],*indices]
    for i in ids:
        states=scene["tracks"][i]["states"][k+1:stop]
        if len(states)!=cfg.future_steps or not all(_valid(s) for s in states):
            raise ValueError("future_censored")
    ego=scene["tracks"][scene["sdc_track_index"]]["states"]
    minimum=100.
    first={d:None for d in cfg.thresholds_m}
    aa=[]; bb=[]
    for step in range(1,cfg.future_steps+1):
        for i in indices:
            aa.append(ego[k+step]); bb.append(scene["tracks"][i]["states"][k+step])
    distances=clearance_many(aa,bb).reshape(cfg.future_steps,len(indices)).min(axis=1)
    minimum=float(distances.min())
    for d in first:
        hits=np.flatnonzero(distances<=d)
        if len(hits): first[d]=float((hits[0]+1)*cfg.dt)
    out={"future_min_clearance":minimum}
    for d in cfg.thresholds_m:
        key=f"{d:g}m"
        eligible=features["current_min_clearance"]>d
        out[f"eligible_{key}"]=int(eligible)
        out[f"label_{key}"]=int(minimum<=d)
        out[f"event_time_{key}"]=first[d]
    return out


def feature_columns(columns,ablation="full"):
    # Explicit prefixes: no metadata, future validity, labels or target times.
    keys=[c for c in columns if c.startswith(("ego_","n")) and (c.startswith("ego_") or (len(c)>1 and c[1].isdigit()))]
    if ablation!="ego":
        keys += [c for c in columns if c in ("neighbor_count","pedestrian_count","cyclist_count","current_min_clearance","cv_min_clearance","cv_time_to_2m")]
    else:
        keys=[c for c in keys if c.startswith("ego_")]
    if ablation=="full":
        keys += [c for c in columns if c.startswith("map_")]
    return sorted(set(keys))
