"""Convenience function to generate all six plots at once."""

from pathlib import Path
from typing import Any

from meshflow.viz.contention import route_contention
from meshflow.viz.heatmap import pe_heatmap
from meshflow.viz.latency import operator_latency
from meshflow.viz.queue import queue_depth
from meshflow.viz.sram import sram_usage
from meshflow.viz.timeline import event_timeline


def dump_all(
    result: Any,
    program: Any,
    output_dir: Path = Path("artifacts/traces"),
) -> list[Path]:
    """Generate all six plots into *output_dir*.

    Extracts ``mesh_width`` and ``mesh_height`` from
    ``program.mesh_config``.  Returns a list of output paths.

    ``result`` is a :class:`meshflow._mesh_runtime.SimResult`.
    ``program`` is a :class:`meshflow.compiler.artifact.RuntimeProgram`.
    """
    w = program.mesh_config.width
    h = program.mesh_config.height

    return [
        pe_heatmap(result, w, h, output_path=output_dir / "pe_heatmap.png"),
        sram_usage(program, output_path=output_dir / "sram_usage.png"),
        route_contention(result, w, h, output_path=output_dir / "route_contention.png"),
        operator_latency(result, output_path=output_dir / "operator_latency.png"),
        queue_depth(result, output_path=output_dir / "queue_depth.png"),
        event_timeline(result, output_path=output_dir / "event_timeline.png"),
    ]
