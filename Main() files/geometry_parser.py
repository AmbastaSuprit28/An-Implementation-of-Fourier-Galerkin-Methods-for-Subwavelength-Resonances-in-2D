from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


def load_geometry(filename: str | Path) -> list[dict]:
    """Load resonator boundaries from the simple DOMAIN/POINTS text format."""
    path = Path(filename)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    domains: list[dict] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("DOMAIN"):
            i += 1
            continue

        if i + 2 >= len(lines) or not lines[i + 1].startswith("CONTRAST"):
            raise ValueError(f"DOMAIN block at line {i + 1} is missing CONTRAST")

        contrast = float(lines[i + 1].split()[1])
        if lines[i + 2] != "POINTS":
            raise ValueError(f"DOMAIN block at line {i + 1} is missing POINTS")

        i += 3
        point_lines = []
        while i < len(lines):
            point_lines.append(lines[i])
            if lines[i] == "]":
                break
            i += 1

        boundary = np.asarray(ast.literal_eval("\n".join(point_lines)), dtype=float)
        if boundary.ndim != 2 or boundary.shape[1] != 2:
            raise ValueError("Each boundary must be an array of [x, y] points")
        if len(boundary) < 8:
            raise ValueError("Each boundary needs at least 8 points for quadrature")

        domains.append({"contrast": contrast, "boundary": boundary})
        i += 1

    if not domains:
        raise ValueError(f"No DOMAIN blocks found in {path}")
    return domains


def save_geometry(filename: str | Path, domains: list[dict]) -> None:
    """Write geometry in the same text format accepted by load_geometry."""
    path = Path(filename)
    chunks = [f"N_DOMAINS {len(domains)}", ""]
    for idx, domain in enumerate(domains, start=1):
        points = np.asarray(domain["boundary"], dtype=float)
        chunks.extend(
            [
                f"DOMAIN {idx}",
                f"CONTRAST {domain['contrast']:.16g}",
                "POINTS",
                "[",
            ]
        )
        chunks.extend(f"[{x:.12g},{y:.12g}]," for x, y in points)
        chunks.extend(["]", ""])
    path.write_text("\n".join(chunks))
