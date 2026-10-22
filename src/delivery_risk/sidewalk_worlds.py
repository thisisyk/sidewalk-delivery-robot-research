"""Separated sidewalk/crosswalk delivery-robot worlds, with exact area checks.

This is synthetic map geometry. Motor traffic stays on the motor carriageway;
people and delivery robots use sidewalks and marked crosswalks. Parked cars
remain completely inside the motor carriageway, including their full footprint.
The learned road-car clearance model is not validated for this robot domain.
"""
from __future__ import annotations

from functools import lru_cache
import math

from .urban_worlds import _arc, _map_features, _rect


SIDEWALK_CATALOG = {
    "intersection": {"name": "十字路口 · 斑马线过街", "description": "机器人沿人行道到斑马线过街；汽车只在机动车道，行人沿步道与斑马线。"},
    "left_turn": {"name": "街角左转 · 沿步道转弯", "description": "机器人沿街角人行道圆弧转弯，与迎面行人会合，不沿机动车道纵行。"},
    "blind_corner": {"name": "建筑盲角 · 人行道突现人", "description": "机器人沿步道绕建筑拐角；楼后的行人在进入视线后才能参与模型评分。"},
    "t_junction": {"name": "T 字路口 · 支路过街", "description": "T 字路口两侧人行道和三条斑马线；汽车在主路与支路通行。"},
    "trash": {"name": "垃圾占道 · 许可路径绕行", "description": "垃圾阻挡部分人行道，可经斑马线到对侧步道绕行；机动车道不作为绕行通道。"},
    "construction": {"name": "步道施工 · 围挡窄口", "description": "人行道围挡形成窄口，施工车停在机动车道内；人和机器人不能借机动车道纵行。"},
    "crowd": {"name": "密集人群 · 步道相遇", "description": "多方向行人沿人行道、斑马线交会；汽车仍只在机动车道。"},
    "mixed": {"name": "综合街角 · 多类占道", "description": "建筑遮挡、垃圾与围挡叠加，机动车道内有停车；机器人只沿步道及斑马线尝试前进。"},
}
for _entry in SIDEWALK_CATALOG.values():
    _entry["title"] = _entry["name"]


def _strip(points, width=5.):
    return {"points": points, "width": float(width)}


def _cross(x0, y0, x1, y1):
    return {"points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def _strip_rect(strip):
    a, b = strip["points"]
    half = strip["width"]/2
    if abs(a[1]-b[1]) < 1e-9:
        return (min(a[0], b[0]), a[1]-half, max(a[0], b[0]), a[1]+half)
    if abs(a[0]-b[0]) < 1e-9:
        return (a[0]-half, min(a[1], b[1]), a[0]+half, max(a[1], b[1]))
    raise ValueError("sidewalk_maps_require_axis_aligned_corridors")


def _cross_rect(cross):
    points = cross["points"]
    return (min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points))


@lru_cache(maxsize=32)
def _disjoint_union(rectangles):
    """Partition an axis-aligned rectangular union into disjoint x-strips."""
    xs = sorted({coordinate for rect in rectangles for coordinate in (rect[0], rect[2])})
    result = []
    for x0, x1 in zip(xs, xs[1:]):
        mid = (x0+x1)/2
        intervals = sorted((r[1], r[3]) for r in rectangles if r[0] <= mid <= r[2])
        merged = []
        for y0, y1 in intervals:
            if merged and y0 <= merged[-1][1]+1e-10:
                merged[-1] = (merged[-1][0], max(merged[-1][1], y1))
            else:
                merged.append((y0, y1))
        result.extend((x0, y0, x1, y1) for y0, y1 in merged)
    return tuple(result)


def _domain_rects(world, road=False):
    if road:
        return tuple(tuple(rect) for rect in world["road_rectangles"])
    return tuple(tuple(rect) for rect in world["walkable_rectangles"])


def point_in_walkable(point, world):
    """Membership in sidewalks UNION crosswalks, with flat corridor ends."""
    x, y = point
    return any(x0-1e-9 <= x <= x1+1e-9 and y0-1e-9 <= y <= y1+1e-9
               for x0, y0, x1, y1 in _domain_rects(world))


def _clip(polygon, axis, bound, keep_greater):
    if not polygon:
        return []
    result = []
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        ia = a[axis] >= bound if keep_greater else a[axis] <= bound
        ib = b[axis] >= bound if keep_greater else b[axis] <= bound
        if ia:
            result.append(a)
        if ia != ib:
            t = (bound-a[axis])/(b[axis]-a[axis])
            result.append((a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])))
    return result


def _area(polygon):
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(polygon, polygon[1:]+polygon[:1])))/2 if len(polygon) >= 3 else 0.


def _footprint_inside(body, world, road=False, margin=0.):
    if margin < 0 or not math.isfinite(margin):
        raise ValueError("nonnegative_finite_margin_required")
    length, width = body["length"]+2*margin, body["width"]+2*margin
    if min(length, width) <= 0 or not all(math.isfinite(body[key]) for key in ("x", "y", "length", "width", "heading")):
        return False
    c, s = math.cos(body["heading"]), math.sin(body["heading"])
    polygon = [(body["x"]+dx*c-dy*s, body["y"]+dx*s+dy*c)
               for dx, dy in ((-length/2, -width/2), (length/2, -width/2),
                              (length/2, width/2), (-length/2, width/2))]
    xlo, xhi = min(p[0] for p in polygon), max(p[0] for p in polygon)
    ylo, yhi = min(p[1] for p in polygon), max(p[1] for p in polygon)
    covered = 0.
    for x0, y0, x1, y1 in _domain_rects(world, road):
        if xhi < x0 or xlo > x1 or yhi < y0 or ylo > y1:
            continue
        clipped = _clip(polygon, 0, x0, True)
        clipped = _clip(clipped, 0, x1, False)
        clipped = _clip(clipped, 1, y0, True)
        clipped = _clip(clipped, 1, y1, False)
        covered += _area(clipped)
    # Full polygon AREA, not merely corners: a footprint may straddle a gap
    # even when its corners happen to lie on two disconnected sidewalk strips.
    return abs(covered-length*width) <= max(1e-7, length*width*1e-8)


def footprint_in_walkable(body, world, margin=0.):
    return _footprint_inside(body, world, margin=margin)


def footprint_in_road(body, world, margin=0.):
    return _footprint_inside(body, world, road=True, margin=margin)


def make_sidewalk_world(scenario, domain="robot"):
    if scenario not in SIDEWALK_CATALOG:
        raise ValueError("unsupported_sidewalk_scenario")
    if domain != "robot":
        raise ValueError("sidewalk_demo_requires_robot_domain")
    roads = [_strip([[0., 35.], [100., 35.]], 12.), _strip([[50., 0.], [50., 70.]], 12.)]
    walkways = [_strip([[0., y], [43., y]]) for y in (25.5, 44.5)]
    walkways += [_strip([[57., y], [100., y]]) for y in (25.5, 44.5)]
    walkways += [_strip([[x, 0.], [x, 28.]]) for x in (40.5, 59.5)]
    walkways += [_strip([[x, 42.], [x, 70.]]) for x in (40.5, 59.5)]
    crosswalks = [_cross(38., 23., 43., 47.), _cross(57., 23., 62., 47.),
                  _cross(38., 23., 62., 28.), _cross(38., 42., 62., 47.)]
    east = [[5., 44.5], [95., 44.5]]
    west = [[95., 25.5], [5., 25.5]]
    north = [[59.5, 5.], [59.5, 65.]]
    corner = [[5., 44.5], [36., 44.5]]+_arc(36., 49., 4.5, -math.pi/2, 0)[1:]+[[40.5, 65.]]
    routes = [east, west, north]
    buildings = [_rect(700, "building", 18., 12., 34., 18., "住宅"),
                 _rect(701, "building", 82., 12., 34., 18., "商店"),
                 _rect(702, "building", 18., 58., 34., 18., "办公楼"),
                 _rect(703, "building", 82., 58., 34., 18., "餐厅")]
    obstacles = []
    if scenario in ("left_turn", "blind_corner", "mixed"):
        routes[0] = corner
    if scenario in ("blind_corner", "mixed"):
        buildings = [b for b in buildings if b["id"] != 702]
        buildings += [_rect(704, "building", 34., 54., 6., 12., "遮挡视线的街角建筑"),
                      _rect(705, "building", 14., 58., 25., 18., "住宅")]
    if scenario == "t_junction":
        roads[1] = _strip([[50., 0.], [50., 35.]], 12.)
        walkways = [_strip([[0., 44.5], [100., 44.5]]),
                    _strip([[0., 25.5], [43., 25.5]]), _strip([[57., 25.5], [100., 25.5]]),
                    _strip([[40.5, 0.], [40.5, 28.]]), _strip([[59.5, 0.], [59.5, 28.]])]
        crosswalks = crosswalks[:3]
        turn = [[59.5, 5.], [59.5, 40.]]+_arc(55., 40., 4.5, 0, math.pi/2)[1:]+[[5., 44.5]]
        routes = [east, west, turn]
        buildings = [b for b in buildings if b["id"] not in (702, 703)]
        buildings.append(_rect(706, "building", 50., 59., 96., 20., "T 字路口北侧建筑"))
    elif scenario in ("trash", "construction"):
        roads = roads[:1]
        walkways = [_strip([[0., y], [100., y]]) for y in (25.5, 44.5)]
        crosswalks = [_cross(18., 23., 23., 47.), _cross(70., 23., 75., 47.)]
        via_crosswalk = [[10., 25.5], [17.5, 25.5]]+_arc(17.5, 28.5, 3., -math.pi/2, 0)[1:]
        via_crosswalk += [[20.5, 41.5]]+_arc(23.5, 41.5, 3., math.pi, math.pi/2)[1:]+[[95., 44.5]]
        routes = [east, west, via_crosswalk]
        if scenario == "trash":
            obstacles = [_rect(900, "trash", 49., 44.5, 4., 3., "阻挡人行道的垃圾堆"),
                         _rect(901, "trash", 52., 45.2, 1.5, 1.8, "散落垃圾"),
                         _rect(902, "parked_car", 64., 39.8, 4.8, 2., "机动车道内路边停车")]
        else:
            obstacles = [_rect(910, "barrier", 49., 43., 12., 1., "步道施工围挡"),
                         _rect(911, "barrier", 49., 46., 12., 1., "步道施工围挡"),
                         _rect(912, "parked_car", 32., 39.8, 4.8, 2., "机动车道内施工车"),
                         _rect(913, "trash", 61., 46.2, 2., 1.2, "材料堆")]
    elif scenario == "crowd":
        obstacles = [_rect(920, "parked_car", 77., 39.8, 4.8, 2., "机动车道内路边停车")]
    elif scenario == "mixed":
        obstacles = [_rect(930, "parked_car", 28., 39.8, 4.8, 2., "机动车道内路边停车"),
                     _rect(934, "trash", 28., 44.5, 3.5, 2.5, "阻挡人行道的纸箱堆"),
                     _rect(931, "trash", 40.5, 57., 2., 2.5, "人行道垃圾堆"),
                     _rect(932, "barrier", 42.1, 64., 1., 5., "步道侧面围挡")]
    road_rectangles = _disjoint_union(tuple(_strip_rect(r) for r in roads))
    walkable = _disjoint_union(tuple(_strip_rect(r) for r in walkways)+tuple(_cross_rect(c) for c in crosswalks))
    lane_lines = [[[0., 32.], [100., 32.]], [[100., 38.], [0., 38.]]]
    if scenario not in ("trash", "construction"):
        end = 35. if scenario == "t_junction" else 70.
        lane_lines += [[[47., end], [47., 0.]], [[53., 0.], [53., end]]]
    maps = _map_features(roads, lane_lines, crosswalks, [[37., 28.], [57., 21.]])
    return {"width": 100, "height": 70, "bounds": [0, 0, 100, 70], "units": "metres",
            "scenario": scenario, "domain": domain, **dict(SIDEWALK_CATALOG[scenario]),
            "roads": roads, "walkways": walkways, "crosswalks": crosswalks,
            "routes": routes, "buildings": buildings, "obstacles": obstacles,
            "map_features": maps, "road_rectangles": road_rectangles,
            "walkable_rectangles": walkable, "synthetic": True, "schema": "sidewalk_v1",
            "route_rules": "robots/pedestrians: sidewalk union marked crosswalk; moving cars: road only",
            "parking_rules": "parked and moving cars remain fully within motor carriageway",
            "domain_warning": "Road-car model applied to sidewalk robots: unvalidated domain transfer; not a safety controller"}


def _spec(ident, type_, points, speed, start, period, label):
    return {"id": ident, "type": type_, "points": points, "speed": speed, "start": start, "period": period, "label": label}


def _specs(scenario):
    car = _spec(800, 1, [[3., 32.], [97., 32.]], 4., 10., 38., "机动车道直行汽车")
    pedestrian = _spec(801, 2, [[40.5, 65.], [40.5, 5.]], 1.2, 8., 70., "沿步道和斑马线行走的行人")
    other = _spec(802, 2, [[95., 44.5], [5., 44.5]], 1.2, 3., 94., "人行道迎面行人")
    vertical_car = _spec(803, 1, [[53., 3.], [53., 67.]], 4., 25., 42., "机动车道横向汽车")
    if scenario == "t_junction":
        turning = [[53., 3.], [53., 32.]]+_arc(47., 32., 6., 0, math.pi/2)[1:]+[[3., 38.]]
        return [car, _spec(803, 1, turning, 3.2, 7., 48., "机动车道支路转弯汽车"),
                _spec(801, 2, [[40.5, 5.], [40.5, 44.5]], 1.1, 0., 56., "支路步道过街行人"), other]
    if scenario in ("trash", "construction"):
        return [car, _spec(801, 2, [[5., 25.5], [95., 25.5]], 1.1, 0., 100., "对侧人行道行人"),
                _spec(802, 2, [[72.5, 23.5], [72.5, 46.5]], 1.1, 6., 40., "斑马线过街行人")]
    if scenario == "mixed":
        return [car, vertical_car,
                _spec(801, 2, [[59.5, 65.], [59.5, 5.]], 1.2, 8., 70., "右侧步道与斑马线行人"),
                _spec(802, 2, [[95., 25.5], [5., 25.5]], 1.2, 3., 94., "南侧步道行人")]
    base = [car, pedestrian, other, vertical_car]
    if scenario == "crowd":
        base += [_spec(804, 2, [[5., 25.5], [95., 25.5]], 1.3, 0., 90., "南侧步道行人"),
                 _spec(805, 2, [[59.5, 5.], [59.5, 65.]], 1., 4., 80., "东侧过街行人")]
    return base


def sidewalk_actors(t, scenario, seed=42, density=1.0):
    """Analytical current states, stable IDs, no future fields or bike traffic."""
    if scenario not in SIDEWALK_CATALOG or not math.isfinite(t) or not 0 <= density <= 3:
        raise ValueError("unsupported_sidewalk_actor_settings")
    if t < 0 or density == 0:
        return []
    specs, actors = _specs(scenario), []
    for index in range(math.ceil(len(specs)*density)):
        spec, repeat = specs[index % len(specs)], index // len(specs)
        start = spec["start"]+((int(seed)*17+spec["id"]*13) % 19)/19*2+repeat*11.
        if t < start:
            continue
        distance = ((t-start) % spec["period"])*spec["speed"]
        segments = [(a, b, math.dist(a, b)) for a, b in zip(spec["points"], spec["points"][1:])]
        if distance > sum(segment[2] for segment in segments):
            continue
        for a, b, length in segments:
            if distance <= length:
                u = distance/length
                x, y = a[0]+u*(b[0]-a[0]), a[1]+u*(b[1]-a[1])
                heading = math.atan2(b[1]-a[1], b[0]-a[0])
                break
            distance -= length
        type_ = spec["type"]
        length, width, kind = (4.5, 1.8, "car") if type_ == 1 else (.65, .65, "pedestrian")
        actors.append({"id": spec["id"]+repeat*1000, "type": type_, "kind": kind,
                       "label": spec["label"], "x": x, "y": y, "heading": heading,
                       "vx": math.cos(heading)*spec["speed"], "vy": math.sin(heading)*spec["speed"],
                       "length": length, "width": width, "valid": True})
    return actors
