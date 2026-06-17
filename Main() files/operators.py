from __future__ import annotations

import numpy as np

EULER_GAMMA = 0.5772156649015328606
K2_CONSTANT = (EULER_GAMMA - 0.5) / (4.0 * np.pi) - 0.125j


def pairwise_dot_normal(target: dict, source: dict) -> tuple[np.ndarray, np.ndarray]:
    x = target["nodes"][:, None, :]
    y = source["nodes"][None, :, :]
    r = x - y
    dot = np.einsum("pqd,pd->pq", r, target["normals"])
    dist = np.linalg.norm(r, axis=2)
    return dot, dist


def k0_prime_kernel(target: dict, source: dict) -> np.ndarray:
    """Kernel for K'_0 using the sign convention needed by C0 = -1/2 I + K'_0."""
    dot, dist = pairwise_dot_normal(target, source)
    kernel = np.zeros_like(dist, dtype=float)
    mask = dist > 1e-13
    kernel[mask] = dot[mask] / (2.0 * np.pi * dist[mask] ** 2)

    if target is source:
        np.fill_diagonal(kernel, target["curvature"] / (4.0 * np.pi))
    return kernel


def k1_prime_kernel(target: dict, source: dict) -> np.ndarray:
    dot, _ = pairwise_dot_normal(target, source)
    kernel = dot / (4.0 * np.pi)
    if target is source:
        np.fill_diagonal(kernel, 0.0)
    return kernel


def k2_prime_kernel(target: dict, source: dict) -> np.ndarray:
    dot, dist = pairwise_dot_normal(target, source)
    kernel = np.zeros_like(dist, dtype=complex)
    mask = dist > 1e-13
    kernel[mask] = dot[mask] * np.log(dist[mask] / 2.0) / (4.0 * np.pi)
    kernel += K2_CONSTANT * dot
    if target is source:
        np.fill_diagonal(kernel, 0.0)
    return kernel
