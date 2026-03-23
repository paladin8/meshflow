"""SRAM usage bar chart."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def sram_usage(
    program: Any,
    output_path: Path = Path("artifacts/traces/sram_usage.png"),
) -> Path:
    """Horizontal bar chart of total floats in initial SRAM per PE.

    ``program`` is a :class:`meshflow.compiler.artifact.RuntimeProgram`.
    """
    labels: list[str] = []
    values: list[int] = []
    for pe in program.pe_programs:  # type: ignore[union-attr]
        labels.append(f"({pe.coord[0]},{pe.coord[1]})")
        values.append(sum(len(v) for v in pe.initial_sram.values()))

    fig, ax = plt.subplots(figsize=(6, max(3, len(labels) * 0.4)))
    ax.barh(labels, values, color="steelblue")
    ax.set_xlabel("Total floats in SRAM")
    ax.set_title("Initial SRAM usage per PE")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path
