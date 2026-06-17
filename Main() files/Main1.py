from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boundary import preprocess_domains
from geometry_parser import load_geometry
from reduced_operator import (
    assemble_effective_matrices,
    build_Ceff,
)
from solver import solve_effective_resonances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 2D subwavelength resonances from boundary coordinates."
    )

    parser.add_argument(
        "geometry",
        nargs="?",
        default="geometry.txt"
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Override geometry contrasts"
    )

    return parser.parse_args()


def plot_geometry(domains):

    plt.figure(figsize=(6, 6))

    for i, dom in enumerate(domains):

        pts = dom["raw_boundary"]

        x = pts[:, 0]
        y = pts[:, 1]

        plt.plot(
            np.r_[x, x[0]],
            np.r_[y, y[0]],
            linewidth=2,
            label=f"Resonator {i+1}"
        )

        plt.fill(
            np.r_[x, x[0]],
            np.r_[y, y[0]],
            alpha=0.25
        )

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Metamaterial Geometry")

    plt.grid(True)
    plt.legend()

    plt.savefig(
        "geometry.png",
        dpi=300,
        bbox_inches="tight"
    )


def plot_resonance_search(K1, K2, delta, roots):

    omegas = np.logspace(-6, -1, 500)

    sigma = []

    for omega in omegas:

        C = build_Ceff(
            K1,
            K2,
            delta,
            omega
        )

        sigma.append(
            np.linalg.svd(
                C,
                compute_uv=False
            )[-1]
        )

    plt.figure(figsize=(8,5))

    plt.loglog(
        omegas,
        sigma,
        linewidth=2,
        label=r"$\sigma_{\min}(C_{\mathrm{eff}})$"
    )

    for root in roots:

        plt.axvline(
            abs(root["omega"]),
            linestyle="--",
            alpha=0.7
        )

    plt.xlabel(r"$\omega$")
    plt.ylabel(
        r"$\sigma_{\min}(C_{\mathrm{eff}})$"
    )

    plt.title(
        "Effective Resonance Search"
    )

    plt.grid(True)
    plt.legend()

    plt.savefig(
        "resonance_search.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
def plot_complex_resonance_map(
    K1,
    K2,
    delta,
    roots
):

    re = np.linspace(
        -0.01,
        0.02,
        250
    )

    im = np.linspace(
        -0.01,
        0.002,
        250
    )

    Z = np.zeros(
        (len(im), len(re))
    )

    for i, yi in enumerate(im):

        for j, xr in enumerate(re):

            omega = xr + 1j * yi

            try:

                C = build_Ceff(
                    K1,
                    K2,
                    delta,
                    omega
                )

                Z[i, j] = np.log10(
                    np.linalg.svd(
                        C,
                        compute_uv=False
                    )[-1]
                    + 1e-30
                )

            except Exception:

                Z[i, j] = np.nan

    plt.figure(figsize=(8,6))

    plt.imshow(
        Z,
        extent=[
            re[0],
            re[-1],
            im[0],
            im[-1]
        ],
        origin="lower",
        aspect="auto"
    )

    plt.colorbar(
        label=r"$\log_{10}\sigma_{\min}(C_{\rm eff})$"
    )

    for k, root in enumerate(roots):

        omega = root["omega"]

        plt.plot(
            omega.real,
            omega.imag,
            "ro",
            markersize=8
        )

        plt.text(
            omega.real,
            omega.imag,
            f"$\\omega_{k+1}$"
        )

    plt.xlabel(
        r"$\Re(\omega)$"
    )

    plt.ylabel(
        r"$\Im(\omega)$"
    )

    plt.title(
        "Complex Resonance Map"
    )

    plt.savefig(
        "complex_resonance_map.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()  

def plot_delta_scaling(K1, K2):

    deltas = np.array([
        1e-8,
        1e-7,
        1e-6,
        1e-5
    ])

    omega_mag = []

    for delta in deltas:

        roots = solve_effective_resonances(
            K1,
            K2,
            delta
        )

        omega_mag.append(
            abs(roots[0]["omega"])
        )

    omega_mag = np.array(omega_mag)

    reference = (
        omega_mag[-1]
        * np.sqrt(deltas / deltas[-1])
    )

    plt.figure(figsize=(7, 5))

    plt.loglog(
        deltas,
        omega_mag,
        "o-",
        linewidth=2,
        label="Computed"
    )

    plt.loglog(
        deltas,
        reference,
        "--",
        linewidth=2,
        label=r"$\sqrt{\delta}$"
    )

    plt.xlabel(r"$\delta$")
    plt.ylabel(r"$|\omega|$")

    plt.title(
        "Subwavelength Resonance Scaling"
    )

    plt.grid(True)
    plt.legend()

    plt.savefig(
        "delta_scaling.png",
        dpi=300,
        bbox_inches="tight"
    )


def main():

    args = parse_args()

    domains = preprocess_domains(
        load_geometry(
            Path(args.geometry)
        )
    )

    contrasts = np.array(
        [dom["contrast"] for dom in domains],
        dtype=float
    )

    if args.delta is not None:

        delta = float(args.delta)

    elif np.max(
        np.abs(contrasts - contrasts[0])
    ) <= 1e-15:

        delta = float(
            contrasts[0]
        )

    else:

        delta = contrasts

        print(
            "Warning: contrasts differ."
        )

    print(
        f"Loaded {len(domains)} resonator(s)"
    )

    for idx, dom in enumerate(
        domains,
        start=1
    ):

        print(
            f"  {idx}: "
            f"contrast={dom['contrast']:.6g}, "
            f"area={dom['area']:.6g}, "
            f"perimeter={dom['perimeter']:.6g}, "
            f"nodes={len(dom['nodes'])}"
        )

    print(
        "\nAssembling effective matrices..."
    )

    K1, K2 = assemble_effective_matrices(
        domains
    )

    print(
        "Solving Ceff(ω,δ)x=0..."
    )

    roots = solve_effective_resonances(
        K1,
        K2,
        delta
    )

    print()
    print("Resonant frequencies")
    print("--------------------")

    for idx, root in enumerate(
        roots,
        start=1
    ):

        omega = root["omega"]

        print(
            f"{idx:2d}: "
            f"omega = "
            f"{omega.real:+.12e} "
            f"{omega.imag:+.12e}j, "
            f"residual = "
            f"{root['residual']:.3e}"
        )

    print("\nGenerating figures...")

    plot_geometry(domains)

    plot_resonance_search(
        K1,
        K2,
        delta,
        roots
    )
    plot_complex_resonance_map(
    K1,
    K2,
    delta,
    roots
)

    plot_delta_scaling(
        K1,
        K2
    )
print("Saved:")
print("  geometry.png")
print("  resonance_search.png")
print("  complex_resonance_map.png")
print("  delta_scaling.png")
   
if __name__ == "__main__":
    main()
