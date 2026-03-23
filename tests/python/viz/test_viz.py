"""Tests for visualization helpers."""

from pathlib import Path

import numpy as np

from meshflow._mesh_runtime import run_program
from meshflow.compiler import compile
from meshflow.compiler.artifact import RuntimeProgram, serialize
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.viz.contention import route_contention
from meshflow.viz.dump import dump_all
from meshflow.viz.heatmap import pe_heatmap
from meshflow.viz.latency import operator_latency
from meshflow.viz.queue import queue_depth
from meshflow.viz.sram import sram_usage
from meshflow.viz.timeline import event_timeline


def _run_simple() -> tuple[object, RuntimeProgram]:
    """FORWARD -> COLLECT: minimal graph."""
    graph = GraphIR(
        nodes=[
            Node(id="a", op=OpType.FORWARD),
            Node(id="b", op=OpType.COLLECT),
        ],
        edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
    )
    program = compile(graph)
    result = run_program(serialize(program), {"a": [1.0, 2.0]})
    return result, program


def _run_linear() -> tuple[object, RuntimeProgram]:
    """LINEAR -> COLLECT: richer profiling data."""
    graph = GraphIR(
        nodes=[
            Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 2, "out_features": 3}),
            Node(id="c", op=OpType.COLLECT),
        ],
        edges=[Edge(src_node="l1", src_slot=0, dst_node="c", dst_slot=0)],
    )
    weights = {
        "l1": {
            "weight": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            "bias": np.array([0.0, 0.0, 0.0]),
        },
    }
    program = compile(graph, weights=weights)
    result = run_program(serialize(program), {"l1": [1.0, 2.0]})
    return result, program


class TestPeHeatmap:
    def test_produces_png(self, tmp_path: Path) -> None:
        result, program = _run_linear()
        out = pe_heatmap(
            result,
            program.mesh_config.width,
            program.mesh_config.height,
            output_path=tmp_path / "heatmap.png",
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_all_metrics(self, tmp_path: Path) -> None:
        result, program = _run_linear()
        for metric in [
            "messages_received",
            "messages_sent",
            "tasks_executed",
            "slots_written",
            "max_queue_depth",
        ]:
            out = pe_heatmap(
                result,
                program.mesh_config.width,
                program.mesh_config.height,
                metric=metric,
                output_path=tmp_path / f"heatmap_{metric}.png",
            )
            assert out.exists()

    def test_simple_graph(self, tmp_path: Path) -> None:
        result, program = _run_simple()
        out = pe_heatmap(
            result,
            program.mesh_config.width,
            program.mesh_config.height,
            output_path=tmp_path / "heatmap_simple.png",
        )
        assert out.exists()
        assert out.stat().st_size > 0


class TestSramUsage:
    def test_produces_png(self, tmp_path: Path) -> None:
        _, program = _run_linear()
        out = sram_usage(program, output_path=tmp_path / "sram.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_simple_graph(self, tmp_path: Path) -> None:
        _, program = _run_simple()
        out = sram_usage(program, output_path=tmp_path / "sram_simple.png")
        assert out.exists()
        assert out.stat().st_size > 0


class TestRouteContention:
    def test_produces_png(self, tmp_path: Path) -> None:
        result, program = _run_linear()
        out = route_contention(
            result,
            program.mesh_config.width,
            program.mesh_config.height,
            output_path=tmp_path / "contention.png",
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_simple_graph(self, tmp_path: Path) -> None:
        result, program = _run_simple()
        out = route_contention(
            result,
            program.mesh_config.width,
            program.mesh_config.height,
            output_path=tmp_path / "contention_simple.png",
        )
        assert out.exists()
        assert out.stat().st_size > 0


class TestOperatorLatency:
    def test_produces_png(self, tmp_path: Path) -> None:
        result, _ = _run_linear()
        out = operator_latency(result, output_path=tmp_path / "latency.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_simple_graph(self, tmp_path: Path) -> None:
        result, _ = _run_simple()
        out = operator_latency(result, output_path=tmp_path / "latency_simple.png")
        assert out.exists()
        assert out.stat().st_size > 0


class TestQueueDepth:
    def test_produces_png(self, tmp_path: Path) -> None:
        result, _ = _run_linear()
        out = queue_depth(result, output_path=tmp_path / "queue.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_simple_graph(self, tmp_path: Path) -> None:
        result, _ = _run_simple()
        out = queue_depth(result, output_path=tmp_path / "queue_simple.png")
        assert out.exists()
        assert out.stat().st_size > 0


class TestEventTimeline:
    def test_produces_png(self, tmp_path: Path) -> None:
        result, _ = _run_linear()
        out = event_timeline(result, output_path=tmp_path / "timeline.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_simple_graph(self, tmp_path: Path) -> None:
        result, _ = _run_simple()
        out = event_timeline(result, output_path=tmp_path / "timeline_simple.png")
        assert out.exists()
        assert out.stat().st_size > 0


class TestDumpAll:
    def test_produces_all_six(self, tmp_path: Path) -> None:
        result, program = _run_linear()
        paths = dump_all(result, program, output_dir=tmp_path)
        assert len(paths) == 6
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_simple_graph(self, tmp_path: Path) -> None:
        result, program = _run_simple()
        paths = dump_all(result, program, output_dir=tmp_path)
        assert len(paths) == 6
        for p in paths:
            assert p.exists()
