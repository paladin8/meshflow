"""PE utilization heatmap."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Valid metric names that can be read from PeStats.
_METRICS = frozenset(
    {"messages_received", "messages_sent", "tasks_executed", "slots_written", "max_queue_depth"}
)


def pe_heatmap(
    result: Any,
    mesh_width: int,
    mesh_height: int,
    metric: str = "tasks_executed",
    output_path: Path = Path("artifacts/traces/pe_heatmap.png"),
) -> Path:
    """2D grid colored by a per-PE metric.

    ``result`` is a :class:`meshflow._mesh_runtime.SimResult`.

    ``metric`` is one of: ``"messages_received"``, ``"messages_sent"``,
    ``"tasks_executed"``, ``"slots_written"``, ``"max_queue_depth"``.
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}, expected one of {sorted(_METRICS)}")

    grid = np.zeros((mesh_height, mesh_width), dtype=np.float64)
    for coord, stats in result.pe_stats.items():  # type: ignore[union-attr]
        x, y = coord
        grid[y, x] = getattr(stats, metric)

    fig, ax = plt.subplots(figsize=(max(4, mesh_width), max(3, mesh_height)))
    im = ax.imshow(grid, origin="lower", cmap="YlOrRd")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"PE heatmap: {metric}")
    ax.set_xticks(range(mesh_width))
    ax.set_yticks(range(mesh_height))
    fig.colorbar(im, ax=ax, label=metric)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path
