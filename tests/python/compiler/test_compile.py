"""Tests for the compile orchestrator."""

import numpy as np
import pytest
from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.artifact import ConcatCollectTask, LinearTask
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType


class TestCompileOrchestrator:
    def test_compile_default_config(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="c", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="c", dst_slot=0),
            ],
        )
        program = compile(graph)

        assert program.version == 1
        assert program.mesh_config.width == 3
        assert program.mesh_config.height == 1
        assert len(program.pe_programs) == 3
        assert len(program.input_slots) == 1
        assert program.input_slots[0].name == "a"

        # Check tasks are correct
        a_prog = next(pe for pe in program.pe_programs if pe.coord == (0, 0))
        assert a_prog.tasks[0].kind == "forward_activation"
        assert a_prog.tasks[0].route_hops == ["east"]

        b_prog = next(pe for pe in program.pe_programs if pe.coord == (1, 0))
        assert b_prog.tasks[0].kind == "forward_activation"
        assert b_prog.tasks[0].route_hops == ["east"]

        c_prog = next(pe for pe in program.pe_programs if pe.coord == (2, 0))
        assert c_prog.tasks[0].kind == "collect_output"
        assert not hasattr(c_prog.tasks[0], "route_hops")

    def test_compile_with_explicit_config(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
        )
        config = CompilerConfig(mesh_width=4, mesh_height=4)
        program = compile(graph, config)

        assert program.mesh_config.width == 4
        assert program.mesh_config.height == 4


class TestLinearCompileOrchestrator:
    def test_compile_linear(self) -> None:
        rng = np.random.default_rng(42)
        W = rng.standard_normal((6, 4)).astype(np.float32)
        b = rng.standard_normal(6).astype(np.float32)

        graph = GraphIR(
            nodes=[
                Node(
                    id="linear1",
                    op=OpType.LINEAR,
                    attrs={"in_features": 4, "out_features": 6},
                )
            ],
            edges=[],
        )
        config = CompilerConfig(mesh_height=4)
        program = compile(graph, config, weights={"linear1": {"weight": W, "bias": b}})

        assert program.version == 1
        assert program.mesh_config.width == 1
        assert program.mesh_config.height == 4
        assert len(program.pe_programs) == 4

        # Tile PEs have LinearTask (vertical layout) — find by task type
        tile_pes = [
            pe for pe in program.pe_programs if any(isinstance(t, LinearTask) for t in pe.tasks)
        ]
        assert len(tile_pes) == 3
        for pe in tile_pes:
            assert len(pe.tasks) == 1
            assert isinstance(pe.tasks[0], LinearTask)
            assert pe.tasks[0].tile_rows == 2
            assert pe.tasks[0].tile_cols == 4
            assert len(pe.initial_sram) == 2  # weight + bias

        # Verify fragment indexing
        fragment_slots = sorted(pe.tasks[0].fragment_slot for pe in tile_pes)
        assert fragment_slots == [0, 1, 2]
        for pe in tile_pes:
            task = pe.tasks[0]
            assert task.fragment_offset == task.fragment_slot * 2

        # Collect PE has ConcatCollectTask entries — find dynamically
        collect_pe = next(
            pe
            for pe in program.pe_programs
            if any(isinstance(t, ConcatCollectTask) for t in pe.tasks)
        )
        assert len(collect_pe.tasks) == 3
        for task in collect_pe.tasks:
            assert isinstance(task, ConcatCollectTask)
            assert task.num_fragments == 3
            assert task.total_rows == 6

        # 3 broadcast input slots
        assert len(program.input_slots) == 3
        assert all(s.name == "linear1" for s in program.input_slots)

    def test_compile_linear_missing_weights(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(
                    id="linear1",
                    op=OpType.LINEAR,
                    attrs={"in_features": 4, "out_features": 6},
                )
            ],
            edges=[],
        )
        with pytest.raises(ValueError, match="requires weights"):
            compile(graph)

    def test_compile_linear_wrong_weight_shape(self) -> None:
        rng = np.random.default_rng(42)
        graph = GraphIR(
            nodes=[
                Node(
                    id="linear1",
                    op=OpType.LINEAR,
                    attrs={"in_features": 4, "out_features": 6},
                )
            ],
            edges=[],
        )
        with pytest.raises(ValueError, match="weight shape"):
            compile(
                graph,
                weights={
                    "linear1": {
                        "weight": rng.standard_normal((3, 4)).astype(np.float32),
                        "bias": rng.standard_normal(6).astype(np.float32),
                    }
                },
            )


class TestShapeChaining:
    def test_shape_mismatch_rejected(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 8}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            compile(
                graph,
                weights={
                    "l1": {
                        "weight": np.zeros((8, 4), dtype=np.float32),
                        "bias": np.zeros(8, dtype=np.float32),
                    },
                    "l2": {
                        "weight": np.zeros((3, 6), dtype=np.float32),
                        "bias": np.zeros(3, dtype=np.float32),
                    },
                },
            )

    def test_shape_mismatch_direct_linear(self) -> None:
        """Direct LINEAR → LINEAR shape mismatch (no RELU between)."""
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 8}),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[Edge(src_node="l1", src_slot=0, dst_node="l2", dst_slot=0)],
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            compile(
                graph,
                weights={
                    "l1": {
                        "weight": np.zeros((8, 4), dtype=np.float32),
                        "bias": np.zeros(8, dtype=np.float32),
                    },
                    "l2": {
                        "weight": np.zeros((3, 6), dtype=np.float32),
                        "bias": np.zeros(3, dtype=np.float32),
                    },
                },
            )

    def test_shape_match_accepted(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 8}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 8, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        # Should not raise — shapes match (l1.out=8 == l2.in=8)
        config = CompilerConfig(mesh_height=4)
        compile(
            graph,
            config,
            weights={
                "l1": {
                    "weight": np.zeros((8, 4), dtype=np.float32),
                    "bias": np.zeros(8, dtype=np.float32),
                },
                "l2": {
                    "weight": np.zeros((3, 8), dtype=np.float32),
                    "bias": np.zeros(3, dtype=np.float32),
                },
            },
        )
