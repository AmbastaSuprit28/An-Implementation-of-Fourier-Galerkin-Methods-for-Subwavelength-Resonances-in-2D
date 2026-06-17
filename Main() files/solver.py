from __future__ import annotations

import numpy as np
import numpy.linalg as la

from reduced_operator import build_Ceff


def resonance_log(omega: complex) -> complex:
    """Principal logarithm used for the paper's resonance branch.

    The asymptotic formulas take arg(omega) in (-pi/2, 0) for outgoing
    resonances. The code does not force Newton iterates onto that quadrant, but
    centralizes the logarithm so branch choices are explicit and auditable.
    """
    return np.log(omega + 0j)


def lower_half_sqrt(value: complex) -> complex:
    root = np.sqrt(value + 0j)
    if root.imag > 0 or (abs(root.imag) < 1e-14 and root.real < 0):
        root = -root
    return root


def solve_effective_resonances(
    K1: np.ndarray,
    K2: np.ndarray,
    delta: float | np.ndarray,
    *,
    max_iter: int = 80,
    tol: float = 1e-11,
) -> list[dict]:
    """Solve the F=0 nonlinear effective problem by fixed-point eigenvalue iteration.

    The equation is D x = lambda (0.5 log(lambda) K1 + K2) x,
    where lambda = omega^2 and log(omega) = 0.5 log(lambda).
    """
    n = K1.shape[0]
    delta_vec = np.full(n, float(delta)) if np.ndim(delta) == 0 else np.asarray(delta, dtype=float)
    D = np.diag(delta_vec).astype(complex)

    scale = max(float(np.mean(delta_vec)), 1e-16)
    initial_log = 0.5 * np.log(scale)
    initial_matrix = K2 + initial_log * K1
    lambdas = _eigvals_pinv_times(initial_matrix, D)

    if len(lambdas) < n:
        lambdas = np.full(n, scale + 0j)

    roots = []
    for start in lambdas[:n]:
        lam = complex(start)
        if abs(lam) < 1e-30:
            lam = scale + 0j

        for _ in range(max_iter):
            log_lam = np.log(lam + 0j)
            matrix = K2 + 0.5 * log_lam * K1
            candidates = _eigvals_pinv_times(matrix, D)
            if len(candidates) == 0:
                break
            new_lam = candidates[np.argmin(np.abs(candidates - lam))]
            if abs(new_lam - lam) <= tol * max(1.0, abs(lam)):
                lam = complex(new_lam)
                break
            lam = 0.5 * lam + 0.5 * complex(new_lam)

        omega = _newton_refine_omega(K1, K2, delta_vec, lower_half_sqrt(lam))
        lam = omega * omega
        residual = la.svd(build_Ceff(K1, K2, delta_vec, omega), compute_uv=False)[-1]
        roots.append({"omega": omega, "lambda": lam, "residual": float(residual)})

    return _unique_roots(roots, n)


def refine_resonance_rf(
    C0F: np.ndarray,
    K1F: np.ndarray,
    K2F: np.ndarray,
    delta: float | np.ndarray,
    omega0: complex,
    branch_vector: np.ndarray | None = None,
    *,
    max_iter: int = 40,
    tol: float = 1e-12,
) -> dict:
    """Refine an effective seed against R_F.

    If branch_vector is supplied, Newton is applied to c*R_F(omega)c. This
    preserves the branch selected by the F=0 effective eigenvector. Otherwise,
    it falls back to singular-vector Newton on the smallest singular pair.
    """
    omega = complex(omega0)
    history = []

    for _ in range(max_iter):
        RF = _build_rf_quiet(C0F, K1F, K2F, delta, omega)
        singular_values = la.svd(RF, compute_uv=False)
        sigma = float(singular_values[-1])
        log_omega = resonance_log(omega)
        dRF = (2.0 * omega * log_omega + omega) * K1F + 2.0 * omega * K2F

        if branch_vector is None:
            U, _, Vh = la.svd(RF)
            u = U[:, -1]
            v = Vh.conj().T[:, -1]
            g = np.vdot(u, RF @ v)
            gp = np.vdot(u, dRF @ v)
        else:
            c = branch_vector / la.norm(branch_vector)
            g = np.vdot(c, RF @ c)
            gp = np.vdot(c, dRF @ c)

        history.append((omega, sigma))

        if abs(gp) < 1e-30:
            break

        step = g / gp
        max_step = 0.5 * max(abs(omega), 1e-12)
        if abs(step) > max_step:
            step *= max_step / abs(step)

        new_omega = omega - step
        if new_omega.imag > 0:
            new_omega = -new_omega

        if abs(new_omega - omega) <= tol * max(1.0, abs(omega)):
            omega = new_omega
            break
        omega = new_omega

    RF = _build_rf_quiet(C0F, K1F, K2F, delta, omega)
    singular_values = la.svd(RF, compute_uv=False)
    return {
        "omega": omega,
        "sigma_min": float(singular_values[-1]),
        "projected_residual": float(abs(np.vdot(branch_vector / la.norm(branch_vector), RF @ (branch_vector / la.norm(branch_vector))))) if branch_vector is not None else None,
        "iterations": len(history),
        "history": history,
    }


def effective_null_vector(K1: np.ndarray, K2: np.ndarray, delta: float | np.ndarray, omega: complex) -> np.ndarray:
    matrix = build_Ceff(K1, K2, delta, omega)
    _, _, Vh = la.svd(matrix)
    return Vh.conj().T[:, -1]


def embed_constant_mode(vector: np.ndarray, num_modes: int) -> np.ndarray:
    """Embed an N-vector into the n=0 slot of an N*(2F+1) Fourier vector."""
    n_domains = len(vector)
    out = np.zeros(n_domains * num_modes, dtype=complex)
    zero_mode_index = num_modes // 2
    for j, value in enumerate(vector):
        out[j * num_modes + zero_mode_index] = value
    return out


def _eigvals_pinv_times(matrix: np.ndarray, D: np.ndarray) -> np.ndarray:
    vals = la.eigvals(la.pinv(matrix, rcond=1e-13) @ D)
    vals = vals[np.isfinite(vals)]
    vals = vals[np.abs(vals) > 1e-30]
    order = np.argsort(np.abs(vals))
    return vals[order]


def _unique_roots(roots: list[dict], n: int) -> list[dict]:
    roots = sorted(roots, key=lambda item: (abs(item["omega"]), item["residual"]))
    unique: list[dict] = []
    for root in roots:
        if all(abs(root["omega"] - prev["omega"]) > 1e-8 * max(1.0, abs(prev["omega"])) for prev in unique):
            unique.append(root)
        if len(unique) == n:
            break
    return unique


def _newton_refine_omega(
    K1: np.ndarray,
    K2: np.ndarray,
    delta: np.ndarray,
    omega0: complex,
    *,
    max_iter: int = 30,
    tol: float = 1e-13,
) -> complex:
    omega = complex(omega0)
    if abs(omega) < 1e-15:
        return omega

    D = np.diag(delta).astype(complex)
    for _ in range(max_iter):
        matrix = D - omega * omega * np.log(omega) * K1 - omega * omega * K2
        det_value = la.det(matrix)
        log_omega = resonance_log(omega)
        derivative_matrix = -((2.0 * omega * log_omega + omega) * K1 + 2.0 * omega * K2)

        try:
            det_derivative = det_value * np.trace(la.solve(matrix, derivative_matrix))
        except la.LinAlgError:
            det_derivative = _finite_difference_det(K1, K2, D, omega)

        if abs(det_derivative) < 1e-30:
            break

        step = det_value / det_derivative
        max_step = 0.5 * max(abs(omega), 1e-12)
        if abs(step) > max_step:
            step *= max_step / abs(step)

        new_omega = omega - step
        if new_omega.imag > 0:
            new_omega = -new_omega

        if abs(new_omega - omega) <= tol * max(1.0, abs(omega)):
            omega = new_omega
            break
        omega = new_omega
    return omega


def _finite_difference_det(K1: np.ndarray, K2: np.ndarray, D: np.ndarray, omega: complex) -> complex:
    h = 1e-6 * max(abs(omega), 1e-8)
    plus = D - (omega + h) ** 2 * resonance_log(omega + h) * K1 - (omega + h) ** 2 * K2
    minus = D - (omega - h) ** 2 * resonance_log(omega - h) * K1 - (omega - h) ** 2 * K2
    return (la.det(plus) - la.det(minus)) / (2.0 * h)


def _build_rf_quiet(
    C0F: np.ndarray,
    K1F: np.ndarray,
    K2F: np.ndarray,
    delta: float | np.ndarray,
    omega: complex,
) -> np.ndarray:
    n = C0F.shape[0]
    if np.ndim(delta) == 0:
        delta_value = float(delta)
        delta_matrix = delta_value * np.eye(n, dtype=complex)
        c0_part = (1.0 - delta_value) * C0F
    else:
        delta_vec = np.asarray(delta, dtype=float)
        if len(delta_vec) != n:
            if n % len(delta_vec) != 0:
                raise ValueError("Vector delta must have one entry per resonator or Galerkin mode")
            delta_vec = np.repeat(delta_vec, n // len(delta_vec))
        delta_matrix = np.diag(delta_vec)
        c0_part = (np.eye(n, dtype=complex) - delta_matrix) @ C0F
    return c0_part - delta_matrix + omega * omega * resonance_log(omega) * K1F + omega * omega * K2F
