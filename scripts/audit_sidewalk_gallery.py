"""Independently audit saved sidewalk frames without importing the simulator.

Oriented footprints are clipped against the exported disjoint rectangular
domain partition. This checks whole-body area, not just center/corner points.
Pedestrian motion is measured from the saved time series, never icon animation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def polygon(body):
    c, s = math.cos(body["heading"]), math.sin(body["heading"])
    return [(body["x"]+c*u-s*v, body["y"]+s*u+c*v)
            for u, v in ((-body["length"]/2,-body["width"]/2), (body["length"]/2,-body["width"]/2),
                         (body["length"]/2,body["width"]/2), (-body["length"]/2,body["width"]/2))]


def area(points):
    if len(points)<3:
        return 0.
    return abs(sum(p[0]*q[1]-q[0]*p[1] for p,q in zip(points,points[1:]+points[:1])))/2


def clipped(points, rect):
    for axis, boundary, direction in ((0,rect[0],1),(0,rect[2],-1),(1,rect[1],1),(1,rect[3],-1)):
        output = []
        if not points:
            break
        previous = points[-1]
        previous_inside = direction*(previous[axis]-boundary)>=0
        for point in points:
            inside = direction*(point[axis]-boundary)>=0
            if inside != previous_inside:
                f = (boundary-previous[axis])/(point[axis]-previous[axis])
                output.append((previous[0]+f*(point[0]-previous[0]),previous[1]+f*(point[1]-previous[1])))
            if inside:
                output.append(point)
            previous, previous_inside = point, inside
        points = output
    return points


def covered(body, rectangles):
    points = polygon(body)
    x0,x1 = min(p[0] for p in points),max(p[0] for p in points)
    y0,y1 = min(p[1] for p in points),max(p[1] for p in points)
    total = sum(area(clipped(points, r)) for r in rectangles if r[0]<=x1 and r[2]>=x0 and r[1]<=y1 and r[3]>=y0)
    return abs(total-body["length"]*body["width"]) <= 1e-7


def disjoint(rectangles):
    for i, a in enumerate(rectangles):
        if not (a[0]<a[2] and a[1]<a[3]):
            return False
        for b in rectangles[i+1:]:
            overlap = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            if overlap>1e-10:
                return False
    return True


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gallery",type=Path,default=root/"demo/sidewalk_gallery.js")
    p.add_argument("--output",type=Path,default=root/"results/sidewalk/geometry_motion_audit.json")
    args = p.parse_args()
    raw = args.gallery.read_bytes()
    gallery = json.loads(raw.decode("utf-8").split("=",1)[1].strip().removesuffix(";"))
    rows, errors = [], []
    for name, run in gallery["runs"].items():
        world = run["world"]
        walks, roads = world["walkable_rectangles"],world["road_rectangles"]
        assert disjoint(walks) and disjoint(roads), name
        robot_frames = car_frames = pedestrian_frames = moved_steps = 0
        bad_robot = bad_car = bad_person = 0
        previous, motion = {}, {}
        contacts = 0
        for frame in run["frames"]:
            contacts += len(frame.get("contact_vehicle_ids",[]))
            for robot in frame["vehicles"]:
                assert robot["kind"]=="delivery_robot" and abs(robot["length"]-.9)<1e-9 and abs(robot["width"]-.65)<1e-9
                robot_frames += 1; bad_robot += not covered(robot,walks)
                score = robot["score"]
                assert score is None or 0<=score<=1
            for actor in frame["actors"]:
                if actor["type"]==1:
                    car_frames += 1; bad_car += not covered(actor,roads)
                elif actor["type"]==2:
                    pedestrian_frames += 1; bad_person += not covered(actor,walks)
                    ident = str(actor["id"])
                    motion.setdefault(ident,{"samples":0,"moving_steps":0,"distance_m":0.})["samples"] += 1
                    if ident in previous:
                        old = previous[ident]
                        if frame["t"]-old[0]<=.500001:
                            distance = math.hypot(actor["x"]-old[1],actor["y"]-old[2])
                            motion[ident]["distance_m"] += distance
                            if distance>1e-6:
                                motion[ident]["moving_steps"] += 1; moved_steps += 1
                    previous[ident] = (frame["t"],actor["x"],actor["y"])
        parked = [obj for obj in world["obstacles"] if obj.get("kind")=="parked_car"]
        bad_parked = sum(not covered(obj,roads) for obj in parked)
        assert contacts == run["summary"]["physical_overlap_frames"], name
        if bad_robot or bad_car or bad_person or bad_parked or not motion or any(not m["moving_steps"] for m in motion.values()):
            errors.append(name)
        rows.append({"scenario":name,"robot_frames":robot_frames,"robot_off_walkway":bad_robot,
            "dynamic_car_frames":car_frames,"dynamic_car_off_road":bad_car,"parked_cars":len(parked),"parked_car_off_road":bad_parked,
            "pedestrian_frames":pedestrian_frames,"pedestrian_off_walkway":bad_person,
            "moving_pedestrian_tracks":sum(m["moving_steps"]>0 for m in motion.values()),"pedestrian_moving_steps":moved_steps,
            "minimum_saved_pedestrian_distance_m":min((m["distance_m"] for m in motion.values()),default=0),
            "contact_frame_count_matches_saved_summary":True})
    result = {"status":"pass" if not errors else "fail","gallery_sha256":hashlib.sha256(raw).hexdigest(),
        "independent_implementation":"no simulator/controller imports; polygon area clipping of saved whole footprints",
        "domain_partitions_checked_disjoint":True,"area_tolerance_m2":1e-7,"scenes":rows,"errors":errors,
        "pedestrian_motion_measured_from_coordinates":True,"icon_gait_not_used_as_motion_evidence":True,
        "not_a_continuous_time_or_real_safety_guarantee":True,"user_publication_approval":"pending"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"scenes":len(rows),"robot_frames":sum(r["robot_frames"] for r in rows),"errors":errors}))
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    main()
