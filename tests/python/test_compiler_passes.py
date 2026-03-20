"""Tests for compiler passes: place, route, lower."""

import pytest
from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.compiler.passes.place import place
from meshflow.compiler.passes.route import _generate_route_xy, route
from meshflow.compiler.passes.lower import lower
from meshflow.compiler.schedule_ir import Direction


class TestPlacement:
    def test_sequential_three_nodes(self) -> None:
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
        spatial = place(graph, CompilerConfig())

        assert spatial.width == 3
        assert spatial.height == 1
        coords = {n.id: n.coord for n in spatial.nodes}
        assert coords == {"a": (0, 0), "b": (1, 0), "c": (2, 0)}

    def test_auto_sizing_nx1(self) -> None:
        graph = GraphIR(
            nodes=[Node(id=f"n{i}", op=OpType.FORWARD) for i in range(5)],
            edges=[],
        )
        spatial = place(graph, CompilerConfig())
        assert spatial.width == 5
        assert spatial.height == 1

    def test_explicit_mesh_dimensions(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="c", op=OpType.FORWARD),
                Node(id="d", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="d", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="d", dst_slot=0),
                Edge(src_node="c", src_slot=0, dst_node="d", dst_slot=0),
            ],
        )
        config = CompilerConfig(mesh_width=2, mesh_height=2)
        spatial = place(graph, config)

        assert spatial.width == 2
        assert spatial.height == 2
        coords = {n.id: n.coord for n in spatial.nodes}
        assert coords["a"] == (0, 0)
        assert coords["b"] == (1, 0)
        assert coords["c"] == (0, 1)
        assert coords["d"] == (1, 1)

    def test_too_many_nodes_for_mesh(self) -> None:
        graph = GraphIR(
            nodes=[Node(id=f"n{i}", op=OpType.FORWARD) for i in range(5)],
            edges=[],
        )
        config = CompilerConfig(mesh_width=2, mesh_height=2)
        with pytest.raises(ValueError, match="does not fit"):
            place(graph, config)

    def test_single_node(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="x", op=OpType.COLLECT)],
            edges=[],
        )
        spatial = place(graph, CompilerConfig())
        assert spatial.width == 1
        assert spatial.height == 1
        assert spatial.nodes[0].coord == (0, 0)

    def test_empty_graph(self) -> None:
        graph = GraphIR(nodes=[], edges=[])
        spatial = place(graph, CompilerConfig())
        assert spatial.width == 1
        assert spatial.height == 1
        assert spatial.nodes == []

    def test_edges_preserved(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
        )
        spatial = place(graph, CompilerConfig())
        assert len(spatial.edges) == 1
        assert spatial.edges[0].src_node == "a"
        assert spatial.edges[0].dst_node == "b"

    def test_validates_graph(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="a", op=OpType.FORWARD)],
            edges=[Edge(src_node="a", src_slot=0, dst_node="missing", dst_slot=0)],
        )
        with pytest.raises(ValueError, match="unknown destination node"):
            place(graph, CompilerConfig())


class TestRouting:
    def test_xy_routing_hops(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
        )
        spatial = place(graph, CompilerConfig())
        schedule = route(spatial, CompilerConfig())

        # a at (0,0), b at (1,0) — one hop east
        a_pe = next(pe for pe in schedule.pe_schedules if pe.coord == (0, 0))
        assert len(a_pe.tasks) == 1
        task = a_pe.tasks[0]
        assert task.kind == "forward_activation"
        assert task.route_dest == (1, 0)
        assert task.route_hops == [Direction.EAST]

    def test_multi_hop_routing(self) -> None:
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
        spatial = place(graph, CompilerConfig())
        schedule = route(spatial, CompilerConfig())

        # b at (1,0) forwards to c at (2,0)
        b_pe = next(pe for pe in schedule.pe_schedules if pe.coord == (1, 0))
        task = b_pe.tasks[0]
        assert task.route_hops == [Direction.EAST]

    def test_vertical_routing(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
        )
        config = CompilerConfig(mesh_width=1, mesh_height=2)
        spatial = place(graph, config)
        # a at (0,0), b at (0,1) — one hop north
        schedule = route(spatial, config)

        a_pe = next(pe for pe in schedule.pe_schedules if pe.coord == (0, 0))
        task = a_pe.tasks[0]
        assert task.route_hops == [Direction.NORTH]

    def test_collect_task_no_route(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
        )
        spatial = place(graph, CompilerConfig())
        schedule = route(spatial, CompilerConfig())

        b_pe = next(pe for pe in schedule.pe_schedules if pe.coord == (1, 0))
        assert len(b_pe.tasks) == 1
        task = b_pe.tasks[0]
        assert task.kind == "collect_output"
        assert task.route_dest is None
        assert task.route_hops is None

    def test_input_slots_detected(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="c", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="c", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="c", dst_slot=1),
            ],
        )
        spatial = place(graph, CompilerConfig())
        schedule = route(spatial, CompilerConfig())

        input_names = {s.name for s in schedule.input_slots}
        assert input_names == {"a", "b"}

    def test_slot_convention(self) -> None:
        """All tasks use trigger_slot=0 and input_slot=0 in M2."""
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
        )
        spatial = place(graph, CompilerConfig())
        schedule = route(spatial, CompilerConfig())

        for pe in schedule.pe_schedules:
            for task in pe.tasks:
                assert task.trigger_slot == 0
                assert task.input_slot == 0

    def test_2d_routing(self) -> None:
        """XY routing on a 2D mesh: X first, then Y."""
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="c", op=OpType.FORWARD),
                Node(id="d", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="d", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="d", dst_slot=0),
                Edge(src_node="c", src_slot=0, dst_node="d", dst_slot=0),
            ],
        )
        config = CompilerConfig(mesh_width=2, mesh_height=2)
        spatial = place(graph, config)
        schedule = route(spatial, config)

        # a at (0,0) -> d at (1,1): [East, North]
        a_pe = next(pe for pe in schedule.pe_schedules if pe.coord == (0, 0))
        assert a_pe.tasks[0].route_hops == [Direction.EAST, Direction.NORTH]


class TestGenerateRouteXY:
    def test_same_pe(self) -> None:
        assert _generate_route_xy((0, 0), (0, 0)) == []

    def test_east(self) -> None:
        assert _generate_route_xy((0, 0), (3, 0)) == [Direction.EAST] * 3

    def test_west(self) -> None:
        assert _generate_route_xy((3, 0), (0, 0)) == [Direction.WEST] * 3

    def test_north(self) -> None:
        assert _generate_route_xy((0, 0), (0, 2)) == [Direction.NORTH] * 2

    def test_south(self) -> None:
        assert _generate_route_xy((0, 2), (0, 0)) == [Direction.SOUTH] * 2

    def test_xy_ordering(self) -> None:
        """X movement comes before Y movement."""
        hops = _generate_route_xy((0, 0), (3, 2))
        assert hops == [
            Direction.EAST,
            Direction.EAST,
            Direction.EAST,
            Direction.NORTH,
            Direction.NORTH,
        ]

    def test_reverse(self) -> None:
        hops = _generate_route_xy((3, 2), (0, 0))
        assert hops == [
            Direction.WEST,
            Direction.WEST,
            Direction.WEST,
            Direction.SOUTH,
            Direction.SOUTH,
        ]

    def test_adjacent(self) -> None:
        assert _generate_route_xy((0, 0), (1, 0)) == [Direction.EAST]


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
