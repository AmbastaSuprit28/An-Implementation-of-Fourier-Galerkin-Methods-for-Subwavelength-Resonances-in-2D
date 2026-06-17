from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--cx", type=float, default=0.0)
    parser.add_argument("--cy", type=float, default=0.0)
    parser.add_argument("--r", type=float, default=1.0)
    args = parser.parse_args()

    theta = np.linspace(0.0, 2.0 * np.pi, args.n, endpoint=False)
    for x, y in zip(args.cx + args.r * np.cos(theta), args.cy + args.r * np.sin(theta)):
        print(f"[{x:.12g},{y:.12g}],")


if __name__ == "__main__":
    main()
