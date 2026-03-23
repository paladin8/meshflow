"""Queue depth histogram."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def queue_depth(
    result: Any,
    output_path: Path = Path("artifacts/traces/queue_depth.png"),
) -> Path:
    """Histogram of max queue depth across all PEs.

    X axis is depth, Y axis is PE count.  Shows the distribution of
    buffering pressure across the mesh.

    ``result`` is a :class:`meshflow._mesh_runtime.SimResult`.
    """
    depths = [stats.max_queue_depth for stats in result.pe_stats.values()]

    fig, ax = plt.subplots(figsize=(6, 4))
    if depths and max(depths) > 0:
        max_depth = max(depths)
        bins = range(int(max_depth) + 2)  # one bin per integer value
        ax.hist(depths, bins=bins, color="steelblue", edgecolor="white", align="left")
    else:
        ax.hist(depths, bins=1, color="steelblue", edgecolor="white")
    ax.set_xlabel("Max queue depth")
    ax.set_ylabel("PE count")
    ax.set_title("Queue depth distribution across PEs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path
