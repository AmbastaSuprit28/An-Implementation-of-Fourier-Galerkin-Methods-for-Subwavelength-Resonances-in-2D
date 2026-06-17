from __future__ import annotations

import numpy as np


def fourier_modes(F: int) -> np.ndarray:
    if F < 0:
        raise ValueError("F must be non-negative")
    return np.arange(-F, F + 1, dtype=int)


def build_fourier_basis(theta: np.ndarray, perimeter: float, F: int) -> tuple[np.ndarray, np.ndarray]:
    """Return modes and normalized basis values with shape (num_modes, num_nodes)."""
    modes = fourier_modes(F)
    basis = np.exp(1j * modes[:, None] * theta[None, :]) / np.sqrt(perimeter)
    return modes, basis


def build_domain_bases(domains: list[dict], F: int) -> tuple[np.ndarray, list[np.ndarray]]:
    modes = fourier_modes(F)
    bases = [
        np.exp(1j * modes[:, None] * dom["theta"][None, :]) / np.sqrt(dom["perimeter"])
        for dom in domains
    ]
    return modes, bases
