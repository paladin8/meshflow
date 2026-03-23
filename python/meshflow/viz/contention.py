"""Route contention mesh overlay."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def route_contention(
    result: Any,
    mesh_width: int,
    mesh_height: int,
    output_path: Path = Path("artifacts/traces/route_contention.png"),
) -> Path:
    """Mesh grid with edges colored and weighted by link message counts.

    ``result`` is a :class:`meshflow._mesh_runtime.SimResult`.
    """
    link_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = dict(
        result.link_counts  # type: ignore[union-attr]
    )

    # Collect all edges and their counts
    segments: list[list[tuple[float, float]]] = []
    counts: list[float] = []

    for y in range(mesh_height):
        for x in range(mesh_width):
            # Right neighbor
            if x + 1 < mesh_width:
                fwd = link_counts.get(((x, y), (x + 1, y)), 0)
                rev = link_counts.get(((x + 1, y), (x, y)), 0)
                total = fwd + rev
                segments.append([(x, y), (x + 1, y)])
                counts.append(total)
            # Up neighbor
            if y + 1 < mesh_height:
                fwd = link_counts.get(((x, y), (x, y + 1)), 0)
                rev = link_counts.get(((x, y + 1), (x, y)), 0)
                total = fwd + rev
                segments.append([(x, y), (x, y + 1)])
                counts.append(total)

    fig, ax = plt.subplots(figsize=(max(4, mesh_width), max(3, mesh_height)))

    if segments:
        max_count = max(counts) if max(counts) > 0 else 1
        # Normalize for colormap and linewidth
        normed = [c / max_count for c in counts]
        linewidths = [0.5 + 4.0 * n for n in normed]
        cmap = plt.get_cmap("YlOrRd")
        colors = [cmap(n) if c > 0 else (0.8, 0.8, 0.8, 1.0) for n, c in zip(normed, counts)]

        lc = LineCollection(segments, linewidths=linewidths, colors=colors)
        ax.add_collection(lc)

    # Draw PE nodes
    for y in range(mesh_height):
        for x in range(mesh_width):
            ax.plot(x, y, "o", color="steelblue", markersize=10, zorder=5)

    ax.set_xlim(-0.5, mesh_width - 0.5)
    ax.set_ylim(-0.5, mesh_height - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Route contention (message count per link)")
    ax.set_xticks(range(mesh_width))
    ax.set_yticks(range(mesh_height))

    # Add colorbar if there are edges with traffic
    if segments and max(counts) > 0:
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, max(counts)))
        sm.set_array(np.array(counts))
        fig.colorbar(sm, ax=ax, label="messages")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path
