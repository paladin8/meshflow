"""Event timeline scatter plot."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


# Colors for each event kind.
_KIND_COLORS = {
    "message_deliver": "#1f77b4",  # blue
    "task_execute": "#2ca02c",  # green
    "message_send": "#ff7f0e",  # orange
}


def event_timeline(
    result: Any,
    output_path: Path = Path("artifacts/traces/event_timeline.png"),
) -> Path:
    """Scatter plot of events over logical time.

    X = timestamp, Y = PE index (linearized from coord).
    Color = event kind.

    ``result`` is a :class:`meshflow._mesh_runtime.SimResult`.
    """
    events = result.trace_events

    if not events:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Event timeline (no events)")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Derive mesh width from max x coordinate
    max_x = max(e["coord"][0] for e in events) + 1

    # Group by kind for legend
    by_kind: dict[str, tuple[list[int], list[int]]] = {}
    for event in events:
        kind = event["kind"]
        if kind not in by_kind:
            by_kind[kind] = ([], [])
        ts_list, idx_list = by_kind[kind]
        x, y = event["coord"]
        ts_list.append(event["timestamp"])
        idx_list.append(y * max_x + x)

    max_pe_idx = max(idx for _, indices in by_kind.values() for idx in indices)
    fig, ax = plt.subplots(
        figsize=(min(20, max(8, len(events) * 0.05)), max(4, (max_pe_idx + 1) * 0.5))
    )
    for kind, (timestamps, indices) in sorted(by_kind.items()):
        color = _KIND_COLORS.get(kind, "gray")
        ax.scatter(timestamps, indices, c=color, label=kind, s=20, alpha=0.7)

    ax.set_xlabel("Logical time")
    ax.set_ylabel("PE index (y * width + x)")
    ax.set_title("Event timeline")
    ax.legend(loc="upper right", fontsize="small")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path
