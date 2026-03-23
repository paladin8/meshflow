"""Per-operator latency bar chart."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def operator_latency(
    result: Any,
    output_path: Path = Path("artifacts/traces/operator_latency.png"),
) -> Path:
    """Bar chart of total time per task kind.

    Groups ``operator_timings`` by ``task_kind``, sums ``(end_ts - start_ts)``
    per group.  Since the simulator uses a uniform ``task_base_latency``, this
    effectively shows invocation counts weighted by latency.

    ``result`` is a :class:`meshflow._mesh_runtime.SimResult`.
    """
    totals: dict[str, float] = defaultdict(float)
    for timing in result.operator_timings:
        totals[timing["task_kind"]] += timing["end_ts"] - timing["start_ts"]

    kinds = sorted(totals.keys())
    values = [totals[k] for k in kinds]

    fig, ax = plt.subplots(figsize=(max(4, len(kinds) * 1.2), 4))
    ax.bar(kinds, values, color="steelblue")
    ax.set_xlabel("Task kind")
    ax.set_ylabel("Total time (logical ticks)")
    ax.set_title("Operator latency by task kind")
    if kinds:
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(kinds, rotation=30, ha="right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path
