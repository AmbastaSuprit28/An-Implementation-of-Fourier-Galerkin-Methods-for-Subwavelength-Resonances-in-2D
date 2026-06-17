from __future__ import annotations

import numpy as np


def project_kernel(kernel: np.ndarray, target_basis: np.ndarray, source_basis: np.ndarray, target_weights: np.ndarray, source_weights: np.ndarray) -> np.ndarray:
    """Project a Nyström kernel matrix into Fourier-Galerkin coordinates."""
    weighted_kernel = target_weights[:, None] * kernel * source_weights[None, :]
    return target_basis.conj() @ weighted_kernel @ source_basis.T
