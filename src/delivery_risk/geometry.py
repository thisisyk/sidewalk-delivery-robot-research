"""Exact 2D oriented-rectangle clearance (sampled in time)."""
from __future__ import annotations
import numpy as np


def corners(state: dict) -> np.ndarray:
    values = [state[k] for k in ("x", "y", "length", "width", "heading")]
    if not np.isfinite(values).all() or min(state["length"], state["width"]) <= 0:
        raise ValueError("Invalid finite positive rectangle geometry")
    c, s = np.cos(state["heading"]), np.sin(state["heading"])
    rotation = np.array([[c, -s], [s, c]])
    unit = np.array([[1,1],[-1,1],[-1,-1],[1,-1]])
    return (unit * [state["length"]/2, state["width"]/2]) @ rotation.T + [state["x"], state["y"]]


def clearance(a: dict, b: dict) -> float:
    p, q = corners(a), corners(b)
    edges = np.concatenate([np.roll(p,-1,axis=0)-p, np.roll(q,-1,axis=0)-q])
    axes = np.stack([-edges[:,1], edges[:,0]], axis=1)
    # Separating-axis theorem, including intersecting rectangles without edge crossings.
    sep = False
    for axis in axes:
        u, v = p @ axis, q @ axis
        if u.max() < v.min() or v.max() < u.min():
            sep = True
            break
    if not sep:
        return 0.0
    def point_edges(points, poly):
        starts, ends = poly, np.roll(poly,-1,axis=0)
        d = ends-starts
        delta = points[:,None,:]-starts[None,:,:]
        t = np.clip((delta*d).sum(-1)/(d*d).sum(-1),0,1)
        return np.linalg.norm(delta-t[:,:,None]*d,axis=-1).min()
    return float(min(point_edges(p,q),point_edges(q,p)))


def _batch_corners(states: list[dict]) -> np.ndarray:
    values = np.asarray([[state[key] for key in ("x", "y", "length", "width", "heading")]
                         for state in states], dtype=float)
    if not np.isfinite(values).all() or np.any(values[:, 2:4] <= 0):
        raise ValueError("Invalid finite positive rectangle geometry")
    local = np.array([[1,1],[-1,1],[-1,-1],[1,-1]])[None, :, :] * values[:, None, 2:4] / 2
    c, s = np.cos(values[:, 4, None]), np.sin(values[:, 4, None])
    x = local[:, :, 0] * c - local[:, :, 1] * s + values[:, None, 0]
    y = local[:, :, 0] * s + local[:, :, 1] * c + values[:, None, 1]
    return np.stack([x, y], axis=-1)


def clearance_many(a: list[dict], b: list[dict]) -> np.ndarray:
    """Exact pairwise clearance for equally sized batches, with no all-pairs product.

    Index i compares only a[i] with b[i]. As in :func:`clearance`, touching,
    overlapping and contained rectangles have distance zero. Sampling in time
    remains the caller's responsibility. The scalar implementation is retained
    as an independent reference for tests.
    """
    if len(a) != len(b):
        raise ValueError("Rectangle batches must have equal length")
    if not len(a):
        return np.empty(0, dtype=float)
    p, q = _batch_corners(a), _batch_corners(b)
    # Rectangles have only two distinct edge normals each; opposite edges add
    # no separating axes. Projection axes need not be normalized for SAT.
    pedges, qedges = np.roll(p, -1, axis=1) - p, np.roll(q, -1, axis=1) - q
    edges = np.concatenate([pedges[:, :2], qedges[:, :2]], axis=1)
    axes = np.stack([-edges[:, :, 1], edges[:, :, 0]], axis=-1)
    pp = np.einsum("nvd,nad->nav", p, axes)
    qq = np.einsum("nvd,nad->nav", q, axes)
    separated = np.any((pp.max(axis=2) < qq.min(axis=2)) |
                       (qq.max(axis=2) < pp.min(axis=2)), axis=1)
    result = np.zeros(len(a), dtype=float)
    if not np.any(separated):
        return result
    p, q = p[separated], q[separated]

    def point_edges_squared(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
        edges = np.roll(poly, -1, axis=1) - poly
        delta = points[:, :, None, :] - poly[:, None, :, :]
        denominator = np.sum(edges * edges, axis=-1)[:, None, :]
        projection = np.sum(delta * edges[:, None, :, :], axis=-1) / denominator
        projection = np.clip(projection, 0, 1)
        residual = delta - projection[:, :, :, None] * edges[:, None, :, :]
        return np.sum(residual * residual, axis=-1).min(axis=(1, 2))

    result[separated] = np.sqrt(np.minimum(point_edges_squared(p, q), point_edges_squared(q, p)))
    return result


def extrapolate(state: dict, dt: float) -> dict:
    return {**state,"x":state["x"]+dt*state["vx"],"y":state["y"]+dt*state["vy"]}
