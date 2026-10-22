"""Synthetic urban layouts and analytical *current-state* traffic actors.

Geometry and traffic below are explicit simulator assumptions, not Waymo data.
Only cars, pedestrians and cyclists have the original model's track types.
Buildings, litter and barriers are physical/visibility objects and are never
relabeled as vehicles. Calling ``urban_actors`` at earlier times reconstructs
past states; the scoring adapter must not request future actor states.
"""
from __future__ import annotations

from copy import deepcopy
import math


SCENARIO_CATALOG = {
    "intersection": {"title": "十字路口 · 横穿人车", "description": "十字路口直行配送车、横向来车与斑马线行人；建筑位于四角。"},
    "left_turn": {"title": "左转路口 · 会车与骑行", "description": "主车沿圆弧左转，与对向直行车、横穿行人和骑行者交互。"},
    "blind_corner": {"title": "建筑盲角 · 突现行人", "description": "街角建筑遮挡侧向视线；行人从建筑后走向道路，进入视线后才能用于模型评分。"},
    "t_junction": {"title": "T 字路口 · 汇入与转弯", "description": "支路车辆汇入主路，配送车沿主路通过；含转弯车和支路行人。"},
    "trash": {"title": "垃圾堆堵路 · 停车绕行", "description": "多块垃圾堆与停放车辆阻挡部分车道；垃圾只进入物理避障，不伪装成模型车辆。"},
    "construction": {"title": "施工围挡 · 窄口通行", "description": "两侧施工围挡形成窄口，入口停放车辆缩小可用空间，行人在出口横穿。"},
    "crowd": {"title": "密集人群 · 人车混行", "description": "多方向行人、骑行者与低速车辆交错通过；密度控制增加周期性来流。"},
    "mixed": {"title": "综合街区 · 多类障碍", "description": "建筑遮挡、停车、垃圾堆、围挡、来车和行人共同出现，主车沿转弯路线通行。"},
}
for _catalog_entry in SCENARIO_CATALOG.values():
    _catalog_entry["name"] = _catalog_entry["title"]


def _arc(cx, cy, radius, a0, a1, samples=13):
    return [[round(cx + radius*math.cos(a0+(a1-a0)*i/(samples-1)), 6),
             round(cy + radius*math.sin(a0+(a1-a0)*i/(samples-1)), 6)]
            for i in range(samples)]


def _rect(ident, kind, x, y, length, width, label, heading=0.):
    result = {"id": ident, "kind": kind, "x": float(x), "y": float(y),
              "length": float(length), "width": float(width), "heading": float(heading), "label": label}
    if kind == "parked_car":
        result.update(type=1, vx=0., vy=0., valid=True)
    return result


def _blocks():
    # A 20 m wide road occupies y=25..45 and x=40..60. All these
    # buildings remain outside its footprint, including their full rectangle.
    return [_rect(700, "building", 21, 12, 30, 18, "住宅"),
            _rect(701, "building", 79, 12, 30, 18, "商店"),
            _rect(702, "building", 21, 58, 30, 18, "办公楼"),
            _rect(703, "building", 79, 58, 30, 18, "餐厅")]


def _sample_line(a, b, step=3.):
    length = math.dist(a, b)
    count = max(2, math.ceil(length/step)+1)
    return [[a[0]+(b[0]-a[0])*i/(count-1), a[1]+(b[1]-a[1])*i/(count-1)] for i in range(count)]


def _map_features(roads, routes, crosswalks, stop_signs):
    features = [{"type": "lane", "polyline": route} for route in routes]
    # True synthetic road boundaries, never outlines of buildings/obstacles.
    # At crossings, trim boundary segments inside the other road, so no false
    # road-edge spans straight across the intersecting drivable corridor.
    for road in roads:
        half = road["width"]/2
        for a, b in zip(road["points"], road["points"][1:]):
            dx, dy = b[0]-a[0], b[1]-a[1]
            length = math.hypot(dx, dy)
            nx, ny = -dy/length, dx/length
            for side in (-1, 1):
                start = [a[0]+side*half*nx, a[1]+side*half*ny]
                end = [b[0]+side*half*nx, b[1]+side*half*ny]
                pieces, current = [], []
                for point in _sample_line(start, end, 1.):
                    inside_other = any(other is not road and _distance_to_road(point, other) < other["width"]/2-.01 for other in roads)
                    if inside_other:
                        if len(current) >= 2:
                            pieces.append(current)
                        current = []
                    else:
                        current.append(point)
                if len(current) >= 2:
                    pieces.append(current)
                features.extend({"type": "road_edge", "polyline": piece} for piece in pieces)
    features.extend({"type": "crosswalk", "polygon": crossing["points"]} for crossing in crosswalks)
    features.extend({"type": "stop_sign", "position": point} for point in stop_signs)
    return features


def _distance_to_road(point, road):
    best = math.inf
    for a, b in zip(road["points"], road["points"][1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        u = min(1., max(0., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(dx*dx+dy*dy)))
        best = min(best, math.hypot(point[0]-a[0]-u*dx, point[1]-a[1]-u*dy))
    return best


def make_urban_world(scenario, domain="road"):
    """Return a fresh 100 x 70 metre layout with three valid route starts.

    ``domain`` changes the vehicle in the simulator, not physical map units.
    Route zero is the featured vehicle; subsequent fleet vehicles cycle routes.
    Static parked cars are in ``obstacles`` exactly once. Dynamic actors returned
    by ``urban_actors`` do not repeat any of these static objects.
    """
    if scenario not in SCENARIO_CATALOG:
        raise ValueError("unsupported_urban_scenario")
    if domain not in ("road", "robot"):
        raise ValueError("unsupported_urban_domain")
    roads = [{"points": [[0., 35.], [100., 35.]], "width": 20.},
             {"points": [[50., 0.], [50., 70.]], "width": 20.}]
    east = [[5., 32.], [95., 32.]]
    west = [[95., 38.], [5., 38.]]
    north = [[53., 5.], [53., 65.]]
    left = [[5., 32.], [45., 32.]] + _arc(45, 40, 8, -math.pi/2, 0)[1:] + [[53., 65.]]
    routes = [east, west, north]
    buildings, obstacles = _blocks(), []
    crosswalks = [{"points": [[41., 24.], [45., 24.], [45., 46.], [41., 46.]]},
                  {"points": [[39., 45.], [61., 45.], [61., 49.], [39., 49.]]}]
    stops = [[38., 28.], [57., 23.], [62., 42.]]
    if scenario == "left_turn":
        routes = [left, west, north]
    elif scenario == "blind_corner":
        routes = [left, west, north]
        buildings = [block for block in buildings if block["id"] != 702]
        buildings.append(_rect(704, "building", 35, 54, 22, 22, "遮挡视线的街角商店"))
        # The near building edge x=46/y=43 lies outside the main car's path;
        # its shadow obscures actors at x=48/y>43 until they approach the road.
    elif scenario == "t_junction":
        roads[1]["points"] = [[50., 0.], [50., 35.]]
        turn_west = [[53., 5.], [53., 30.]] + _arc(45, 30, 8, 0, math.pi/2)[1:] + [[5., 38.]]
        turn_south = [[95., 38.], [55., 38.]] + _arc(55, 30, 8, math.pi/2, math.pi)[1:] + [[47., 5.]]
        routes = [east, turn_west, turn_south]
        buildings = [block for block in buildings if block["id"] not in (702, 703)]
        buildings += [_rect(705, "building", 50, 58, 85, 18, "T 字路口对面建筑")]
        crosswalks = [{"points": [[39., 24.], [61., 24.], [61., 28.], [39., 28.]]},
                      {"points": [[63., 25.], [67., 25.], [67., 45.], [63., 45.]]}]
        stops = [[57., 23.]]
    elif scenario in ("trash", "construction"):
        roads = [{"points": [[0., 35.], [100., 35.]], "width": 26.}]
        routes = [[[5., 35.], [95., 35.]], [[5., 28.], [95., 28.]], [[95., 42.], [5., 42.]]]
        buildings = [_rect(710, "building", 25, 10, 40, 16, "仓库"),
                     _rect(711, "building", 76, 10, 38, 16, "商店"),
                     _rect(712, "building", 24, 60, 40, 16, "住宅"),
                     _rect(713, "building", 76, 60, 38, 16, "餐厅")]
        crosswalks = [{"points": [[70., 22.], [74., 22.], [74., 48.], [70., 48.]]}]
        stops = []
        if scenario == "trash":
            obstacles = [_rect(900, "trash", 49, 35, 5, 4, "大件垃圾堆"),
                         _rect(901, "trash", 52.5, 37.5, 2, 2.5, "散落垃圾"),
                         _rect(902, "parked_car", 65, 42, 4.8, 2, "停放车辆")]
        else:
            obstacles = [_rect(910, "barrier", 49, 26, 18, 8, "施工围挡"),
                         _rect(911, "barrier", 49, 45, 18, 8, "施工围挡"),
                         _rect(912, "parked_car", 32, 42, 5, 2, "施工车辆"),
                         _rect(913, "trash", 61, 46, 3.5, 2, "施工材料堆")]
    elif scenario == "crowd":
        # A broad crossing concentrates mixed traffic without an artificial
        # building polygon being used as model map geometry.
        crosswalks += [{"points": [[61., 24.], [65., 24.], [65., 46.], [61., 46.]]}]
        obstacles = [_rect(920, "parked_car", 74, 44, 4.8, 1.9, "路边停放车")]
    elif scenario == "mixed":
        routes = [left, west, north]
        buildings = [block for block in buildings if block["id"] != 702]
        buildings.append(_rect(704, "building", 35, 58, 22, 18, "街角商店"))
        obstacles = [_rect(930, "parked_car", 28, 32, 4.8, 1.9, "违停车辆"),
                     _rect(931, "trash", 59, 56, 3, 3, "垃圾堆"),
                     _rect(932, "barrier", 46, 61, 2, 8, "施工围挡"),
                     _rect(933, "trash", 63, 31, 2, 2, "散落纸箱")]
    return {"width": 100, "height": 70, "bounds": [0, 0, 100, 70],
            "scenario": scenario, "domain": domain, **deepcopy(SCENARIO_CATALOG[scenario]),
            "roads": roads, "routes": routes, "buildings": buildings,
            "obstacles": obstacles, "crosswalks": crosswalks,
            "map_features": _map_features(roads, routes, crosswalks, stops),
            "visibility": "building rectangles occlude line of sight; not learned by original model",
            "synthetic": True}


def _actor(ident, type_, points, speed, start, period, label):
    return {"id": ident, "type": type_, "points": points, "speed": speed,
            "start": start, "period": period, "label": label}


def _actor_specs(scenario):
    # Starts are scenario-time seconds; all complete routes have a blank interval
    # before repeating. Therefore there is no visible teleport between endpoints.
    base = [_actor(800, 1, [[53, 3], [53, 67]], 4., 4., 38., "横向来车"),
            _actor(801, 2, [[43, 23], [43, 47]], 1.25, 5., 36., "斑马线行人"),
            _actor(802, 3, [[97, 41], [3, 41]], 3.5, 12., 44., "骑行者")]
    if scenario == "left_turn":
        return [_actor(800, 1, [[95, 38], [5, 38]], 4., 1., 38., "对向直行车"),
                _actor(801, 2, [[39, 47], [61, 47]], 1.25, 10., 36., "转弯出口行人"),
                _actor(802, 3, [[56, 67], [56, 3]], 3., 7., 38., "对向骑行者")]
    if scenario == "blind_corner":
        return [_actor(801, 2, [[48, 63], [48, 23]], 1.35, 0., 45., "从建筑后出现的行人"),
                _actor(802, 3, [[48, 63], [48, 23]], 2.3, 12., 38., "建筑后的骑行者"),
                _actor(800, 1, [[95, 38], [5, 38]], 3.2, 7., 44., "对向来车")]
    if scenario == "t_junction":
        turn = [[53, 3], [53, 30]]+_arc(45, 30, 8, 0, math.pi/2)[1:]+[[3, 38]]
        return [_actor(800, 1, turn, 3.4, 5., 48., "支路汇入车"),
                _actor(801, 2, [[39, 26], [61, 26]], 1.2, 7., 36., "支路横穿行人"),
                _actor(802, 3, [[97, 41], [3, 41]], 3.3, 2., 44., "主路骑行者")]
    if scenario == "construction":
        return [_actor(801, 2, [[72, 21], [72, 49]], 1.2, 2., 40., "出口横穿行人"),
                _actor(802, 3, [[97, 38.5], [3, 38.5]], 3.2, 5., 46., "窄口骑行者"),
                _actor(803, 1, [[97, 37.5], [3, 37.5]], 3.5, 18., 45., "窄口迎面来车")]
    if scenario == "trash":
        return [_actor(801, 2, [[72, 21], [72, 49]], 1.2, 2., 40., "出口横穿行人"),
                _actor(802, 3, [[97, 45], [3, 45]], 3.2, 5., 46., "路侧骑行者"),
                _actor(803, 1, [[97, 28], [3, 28]], 3.5, 18., 45., "迎面来车")]
    if scenario == "crowd":
        return base + [_actor(803, 2, [[63, 47], [63, 23]], 1.1, 1., 36., "另一侧行人"),
                       _actor(804, 2, [[39, 47], [61, 47]], 1.35, 4., 35., "路口行人"),
                       _actor(805, 3, [[47, 3], [47, 67]], 3., 8., 38., "穿行骑行者")]
    if scenario == "mixed":
        return [_actor(800, 1, [[95, 38], [5, 38]], 3.6, 3., 42., "对向来车"),
                _actor(801, 2, [[48, 65], [48, 23]], 1.3, 0., 48., "街角行人"),
                _actor(802, 3, [[56, 67], [56, 3]], 2.7, 7., 42., "路口骑行者"),
                _actor(803, 2, [[39, 47], [61, 47]], 1.15, 10., 39., "横穿行人")]
    return base


def _path_pose(points, distance):
    for a, b in zip(points, points[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        segment = math.hypot(dx, dy)
        if distance <= segment:
            return a[0]+dx*distance/segment, a[1]+dy*distance/segment, math.atan2(dy, dx)
        distance -= segment
    return points[-1][0], points[-1][1], math.atan2(points[-1][1]-points[-2][1], points[-1][0]-points[-2][0])


def urban_actors(t, scenario, seed=42, density=1.0):
    """Return analytical current actor states, never future trajectories.

    Same arguments/time always give identical states. An actor ID stays stable
    during its passage. Each period has an absent interval, so history adapters
    must mark missing observations invalid rather than extrapolate through gaps.
    ``density`` in [0, 3] varies actor count (zero removes all dynamic traffic).
    Obstacles/parked cars are exclusively in ``make_urban_world``.
    """
    if scenario not in SCENARIO_CATALOG:
        raise ValueError("unsupported_urban_scenario")
    if not math.isfinite(t) or not 0 <= density <= 3:
        raise ValueError("unsupported_urban_actor_settings")
    if t < 0 or density == 0:
        return []
    specs = _actor_specs(scenario)
    count = min(24, int(math.ceil(len(specs)*density)))
    result = []
    for index in range(count):
        spec, repeat = specs[index % len(specs)], index // len(specs)
        # Arithmetic phase jitter is stable, reproducible and independent of any
        # model labels or scores. Extra-density copies arrive at different times.
        jitter = ((int(seed)*17+spec["id"]*13) % 19)/19*2
        start = spec["start"]+jitter+repeat*9.
        elapsed = t-start
        if elapsed < 0:
            continue
        phase = elapsed % spec["period"]
        path_length = sum(math.dist(a, b) for a, b in zip(spec["points"], spec["points"][1:]))
        distance = phase*spec["speed"]
        if distance > path_length:
            continue
        x, y, heading = _path_pose(spec["points"], distance)
        type_ = spec["type"]
        length, width, kind = {1: (4.5, 1.8, "car"), 2: (.65, .65, "pedestrian"), 3: (1.8, .65, "cyclist")}[type_]
        result.append({"id": spec["id"]+repeat*1000, "type": type_, "kind": kind,
                       "label": spec["label"], "x": x, "y": y,
                       "vx": math.cos(heading)*spec["speed"], "vy": math.sin(heading)*spec["speed"],
                       "heading": heading, "length": length, "width": width, "valid": True})
    return result
