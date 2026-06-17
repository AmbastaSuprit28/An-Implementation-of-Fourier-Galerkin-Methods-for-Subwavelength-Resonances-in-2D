from __future__ import annotations

import numpy as np


def _signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _curvature_periodic(nodes: np.ndarray, theta: np.ndarray) -> np.ndarray:
    x = nodes[:, 0]
    y = nodes[:, 1]
    dtheta = float(theta[1] - theta[0]) if len(theta) > 1 else 1.0
    xp = (np.roll(x, -1) - np.roll(x, 1)) / (2.0 * dtheta)
    yp = (np.roll(y, -1) - np.roll(y, 1)) / (2.0 * dtheta)
    xpp = (np.roll(x, -1) - 2.0 * x + np.roll(x, 1)) / (dtheta * dtheta)
    ypp = (np.roll(y, -1) - 2.0 * y + np.roll(y, 1)) / (dtheta * dtheta)
    denom = np.maximum((xp * xp + yp * yp) ** 1.5, 1e-15)
    return (xp * ypp - yp * xpp) / denom


def preprocess_boundary(boundary: np.ndarray) -> dict:
    """Convert ordered boundary vertices into midpoint quadrature data.

    Points are oriented counter-clockwise so that normals are outward.
    The quadrature nodes are segment midpoints; this matches the segment
    weights and avoids mixing vertex values with midpoint weights.
    """
    points = np.asarray(boundary, dtype=float)
    if np.linalg.norm(points[0] - points[-1]) < 1e-12:
        points = points[:-1]

    signed_area = _signed_area(points)
    if signed_area < 0.0:
        points = points[::-1].copy()
        signed_area = -signed_area

    nxt = np.roll(points, -1, axis=0)
    tangent = nxt - points
    ds = np.linalg.norm(tangent, axis=1)
    if np.any(ds <= 1e-14):
        raise ValueError("Boundary contains repeated or nearly repeated points")

    nodes = 0.5 * (points + nxt)
    weights = ds
    perimeter = float(np.sum(weights))
    normals = np.column_stack((tangent[:, 1], -tangent[:, 0])) / ds[:, None]

    s_mid = np.cumsum(weights) - 0.5 * weights
    theta = 2.0 * np.pi * s_mid / perimeter
    curvature = _curvature_periodic(nodes, theta)

    return {
        "raw_boundary": points,
        "boundary": nodes,
        "nodes": nodes,
        "weights": weights,
        "normals": normals,
        "theta": theta,
        "ds": ds,
        "perimeter": perimeter,
        "area": signed_area,
        "curvature": curvature,
    }


def preprocess_domains(domains: list[dict]) -> list[dict]:
    processed = []
    for domain in domains:
        data = preprocess_boundary(domain["boundary"])
        data["contrast"] = float(domain["contrast"])
        processed.append(data)
    return processed
