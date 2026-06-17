from __future__ import annotations

import numpy as np

from fourier import build_domain_bases
from galerkin_projection import project_kernel
from operators import K2_CONSTANT, k0_prime_kernel, k1_prime_kernel, k2_prime_kernel


def resonance_log(omega: complex) -> complex:
    return np.log(omega + 0j)


def _block_slices(domains: list[dict], F: int) -> list[slice]:
    m = 2 * F + 1
    return [slice(i * m, (i + 1) * m) for i in range(len(domains))]


def _assemble_projected(domains: list[dict], bases: list[np.ndarray], kernel_fn) -> np.ndarray:
    F = (bases[0].shape[0] - 1) // 2
    m = 2 * F + 1
    size = len(domains) * m
    out = np.zeros((size, size), dtype=complex)
    slices = _block_slices(domains, F)

    for i, target in enumerate(domains):
        for j, source in enumerate(domains):
            kernel = kernel_fn(target, source)
            block = project_kernel(
                kernel,
                bases[i],
                bases[j],
                target["weights"],
                source["weights"],
            )
            out[slices[i], slices[j]] = block
    return out


def assemble_K0primeF(domains: list[dict], bases: list[np.ndarray]) -> np.ndarray:
    return _assemble_projected(domains, bases, k0_prime_kernel)


def assemble_C0F(domains: list[dict], bases: list[np.ndarray] | None = None, F: int | None = None) -> np.ndarray:
    if bases is None:
        if F is None:
            raise ValueError("Provide either bases or F")
        _, bases = build_domain_bases(domains, F)
    k0 = assemble_K0primeF(domains, bases)
    return k0 - 0.5 * np.eye(k0.shape[0], dtype=complex)


def assemble_K1F(domains: list[dict], bases: list[np.ndarray] | None = None, F: int | None = None) -> np.ndarray:
    if bases is None:
        if F is None:
            raise ValueError("Provide either bases or F")
        _, bases = build_domain_bases(domains, F)
    return _assemble_projected(domains, bases, k1_prime_kernel)


def assemble_K2F(domains: list[dict], bases: list[np.ndarray] | None = None, F: int | None = None) -> np.ndarray:
    if bases is None:
        if F is None:
            raise ValueError("Provide either bases or F")
        _, bases = build_domain_bases(domains, F)
    return _assemble_projected(domains, bases, k2_prime_kernel)


def build_RF(C0F: np.ndarray, K1F: np.ndarray, K2F: np.ndarray, delta: float | np.ndarray, omega: complex) -> np.ndarray:
    """Build R_F(omega, delta) from equation (53)."""
    n = C0F.shape[0]
    if np.ndim(delta) == 0:
        delta_matrix = float(delta) * np.eye(n, dtype=complex)
        c0_factor = 1.0 - float(delta)
        c0_part = c0_factor * C0F
    else:
        delta_vec = np.asarray(delta, dtype=float)
        if len(delta_vec) != n:
            if n % len(delta_vec) != 0:
                raise ValueError("Vector delta must have one entry per resonator or one entry per Galerkin mode")
            delta_vec = np.repeat(delta_vec, n // len(delta_vec))
        delta_matrix = np.diag(delta_vec)
        c0_part = (np.eye(n, dtype=complex) - delta_matrix) @ C0F
    return c0_part - delta_matrix + omega * omega * resonance_log(omega) * K1F + omega * omega * K2F


def assemble_effective_matrices(domains: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the F=0 matrices K1^0 and K2^0 from Proposition 2."""
    n = len(domains)
    K1 = np.zeros((n, n), dtype=complex)
    K2 = np.zeros((n, n), dtype=complex)

    for i, target in enumerate(domains):
        for j, source in enumerate(domains):
            K1[i, j] = (
                target["area"]
                / (2.0 * np.pi)
                * np.sqrt(source["perimeter"] / target["perimeter"])
            )

            dot, dist = _pairwise_dot_and_dist(target, source)
            mask = dist > 1e-13
            log_part = np.zeros_like(dist, dtype=float)
            log_part[mask] = dot[mask] * np.log(dist[mask] / 2.0)
            integral = np.sum(
                log_part * target["weights"][:, None] * source["weights"][None, :]
            )
            K2[i, j] = (
                integral / (4.0 * np.pi * np.sqrt(target["perimeter"] * source["perimeter"]))
                + (4.0 * np.pi * K2_CONSTANT) * K1[i, j]
            )
    return K1, K2


def _pairwise_dot_and_dist(target: dict, source: dict) -> tuple[np.ndarray, np.ndarray]:
    x = target["nodes"][:, None, :]
    y = source["nodes"][None, :, :]
    r = x - y
    dot = np.einsum("pqd,pd->pq", r, target["normals"])
    dist = np.linalg.norm(r, axis=2)
    return dot, dist


def build_Ceff(K1: np.ndarray, K2: np.ndarray, delta: float | np.ndarray, omega: complex) -> np.ndarray:
    n = K1.shape[0]
    if np.ndim(delta) == 0:
        delta_matrix = float(delta) * np.eye(n, dtype=complex)
    else:
        delta_vec = np.asarray(delta, dtype=float)
        if len(delta_vec) != n:
            raise ValueError("Vector delta must have one entry per resonator")
        delta_matrix = np.diag(delta_vec)
    return delta_matrix - omega * omega * resonance_log(omega) * K1 - omega * omega * K2
