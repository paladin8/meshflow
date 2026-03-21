"""Tests for the lowering pass."""

from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.compiler.passes.lower import lower
from meshflow.compiler.passes.place import place
from meshflow.compiler.passes.route import route


class TestLowering:
    def test_lower_simple_chain(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
        )
        spatial = place(graph, CompilerConfig())
        schedule = route(spatial, CompilerConfig())
        program = lower(schedule)

        assert program.version == 1
        assert program.mesh_config.width == 2
        assert program.mesh_config.height == 1
        assert len(program.pe_programs) == 2
        assert len(program.input_slots) == 1
        assert program.input_slots[0].name == "a"

    def test_lower_direction_to_strings(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
        )
        spatial = place(graph, CompilerConfig())
        schedule = route(spatial, CompilerConfig())
        program = lower(schedule)

        a_prog = next(pe for pe in program.pe_programs if pe.coord == (0, 0))
        task = a_prog.tasks[0]
        assert task.route_hops == ["east"]

    def test_lower_empty_initial_sram(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="a", op=OpType.COLLECT)],
            edges=[],
        )
        spatial = place(graph, CompilerConfig())
        schedule = route(spatial, CompilerConfig())
        program = lower(schedule)

        for pe in program.pe_programs:
            assert pe.initial_sram == {}

    def test_lower_defaults(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="a", op=OpType.COLLECT)],
            edges=[],
        )
        program = compile(graph)

        assert program.mesh_config.hop_latency == 1
        assert program.mesh_config.task_base_latency == 1
        assert program.mesh_config.max_events == 100_000
