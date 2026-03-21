"""Tests for compiler passes: place, route, lower."""

import numpy as np
import pytest
from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.artifact import ConcatCollectTask, LinearTask
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
        assert not hasattr(task, "route_dest")
        assert not hasattr(task, "route_hops")

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


class TestLinearPlacement:
    def _make_linear_graph(self, in_f: int = 4, out_f: int = 6) -> GraphIR:
        return GraphIR(
            nodes=[
                Node(
                    id="linear1",
                    op=OpType.LINEAR,
                    attrs={"in_features": in_f, "out_features": out_f},
                )
            ],
            edges=[],
        )

    def test_linear_expands_to_tiles_and_collect(self) -> None:
        graph = self._make_linear_graph(in_f=4, out_f=6)
        config = CompilerConfig(mesh_height=4)
        spatial = place(graph, config)

        # 3 tile PEs + 1 collect = 4 placed nodes, vertical layout
        assert len(spatial.nodes) == 4
        assert spatial.width == 1
        assert spatial.height == 4

        # First 3 are tile PEs (stacked vertically)
        for i in range(3):
            tile = spatial.nodes[i]
            assert tile.id == f"linear1_tile_{i}"
            assert tile.op == OpType.LINEAR
            assert tile.coord == (0, i)
            assert tile.attrs is not None
            assert tile.attrs["tile_index"] == i
            assert tile.attrs["tile_rows"] == 2
            assert tile.attrs["in_features"] == 4
            assert tile.attrs["origin_id"] == "linear1"

        # Last is collect PE at top of column
        collect = spatial.nodes[3]
        assert collect.id == "linear1_collect"
        assert collect.op == OpType.COLLECT
        assert collect.coord == (0, 3)
        assert collect.attrs is not None
        assert collect.attrs["num_fragments"] == 3
        assert collect.attrs["total_rows"] == 6

    def test_linear_generates_internal_edges(self) -> None:
        graph = self._make_linear_graph(in_f=4, out_f=6)
        config = CompilerConfig(mesh_height=4)
        spatial = place(graph, config)

        # 3 edges: tile_0->collect, tile_1->collect, tile_2->collect
        assert len(spatial.edges) == 3
        for i, edge in enumerate(spatial.edges):
            assert edge.src_node == f"linear1_tile_{i}"
            assert edge.dst_node == "linear1_collect"
            assert edge.dst_slot == i

    def test_linear_auto_sizing(self) -> None:
        """Without explicit mesh_height, one PE per output row."""
        graph = self._make_linear_graph(in_f=2, out_f=3)
        spatial = place(graph, CompilerConfig())

        # 3 tiles + 1 collect = 4 nodes, height=4 (vertical)
        assert len(spatial.nodes) == 4
        assert spatial.width == 1
        assert spatial.height == 4

    def test_linear_validation_missing_attrs(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="bad", op=OpType.LINEAR)],
            edges=[],
        )
        with pytest.raises(ValueError, match="requires attrs"):
            place(graph, CompilerConfig())

    def test_linear_uneven_tiling(self) -> None:
        """out_features=7 with 3 tiles uses base/remainder distribution."""
        graph = GraphIR(
            nodes=[
                Node(
                    id="linear1",
                    op=OpType.LINEAR,
                    attrs={"in_features": 4, "out_features": 7},
                )
            ],
            edges=[],
        )
        config = CompilerConfig(mesh_height=4)
        spatial = place(graph, config)

        # 3 tiles: base=2, remainder=1 → tiles get [3, 2, 2] rows
        assert len(spatial.nodes) == 4
        tile0 = spatial.nodes[0]
        tile1 = spatial.nodes[1]
        tile2 = spatial.nodes[2]
        assert tile0.attrs["tile_rows"] == 3
        assert tile0.attrs["fragment_offset"] == 0
        assert tile1.attrs["tile_rows"] == 2
        assert tile1.attrs["fragment_offset"] == 3
        assert tile2.attrs["tile_rows"] == 2
        assert tile2.attrs["fragment_offset"] == 5


class TestLinearRouting:
    def _make_linear_weights(
        self, in_f: int = 4, out_f: int = 6
    ) -> dict[str, dict[str, np.ndarray]]:
        rng = np.random.default_rng(42)
        return {
            "linear1": {
                "weight": rng.standard_normal((out_f, in_f)).astype(np.float32),
                "bias": rng.standard_normal(out_f).astype(np.float32),
            }
        }

    def test_linear_routing_creates_tasks(self) -> None:
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
        weights = self._make_linear_weights()

        spatial = place(graph, config)
        schedule = route(spatial, config, weights)

        # 4 PEs in schedule (vertical layout)
        assert len(schedule.pe_schedules) == 4

        # Tile PEs have linear tasks (stacked vertically)
        for i in range(3):
            pe = next(p for p in schedule.pe_schedules if p.coord == (0, i))
            assert len(pe.tasks) == 1
            task = pe.tasks[0]
            assert task.kind == "linear"
            assert task.weight_slot == 1
            assert task.bias_slot == 2
            assert task.tile_rows == 2
            assert task.tile_cols == 4
            assert task.fragment_slot == i
            assert task.fragment_offset == i * 2
            assert task.route_dest == (0, 3)

        # Collect PE has 3 concat_collect tasks
        collect_pe = next(p for p in schedule.pe_schedules if p.coord == (0, 3))
        assert len(collect_pe.tasks) == 3
        for i, task in enumerate(collect_pe.tasks):
            assert task.kind == "concat_collect"
            assert task.trigger_slot == i
            assert task.num_fragments == 3
            assert task.total_rows == 6
            assert task.fragment_offset == i * 2

    def test_linear_weight_tiling_in_sram(self) -> None:
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
        weights = self._make_linear_weights()

        spatial = place(graph, config)
        schedule = route(spatial, config, weights)

        W = weights["linear1"]["weight"]
        b = weights["linear1"]["bias"]

        for i in range(3):
            pe = next(p for p in schedule.pe_schedules if p.coord == (0, i))
            # Weight tile in slot 1
            expected_w = W[i * 2 : (i + 1) * 2, :].flatten().tolist()
            assert pe.initial_sram[1] == pytest.approx(expected_w)
            # Bias tile in slot 2
            expected_b = b[i * 2 : (i + 1) * 2].flatten().tolist()
            assert pe.initial_sram[2] == pytest.approx(expected_b)

    def test_linear_broadcast_input_slots(self) -> None:
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
        weights = self._make_linear_weights()

        spatial = place(graph, config)
        schedule = route(spatial, config, weights)

        # 3 input slots, all named "linear1" (broadcast)
        assert len(schedule.input_slots) == 3
        for slot in schedule.input_slots:
            assert slot.name == "linear1"
            assert slot.payload_slot == 0


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

        # Tile PEs have LinearTask (vertical layout)
        for i in range(3):
            pe = next(p for p in program.pe_programs if p.coord == (0, i))
            assert len(pe.tasks) == 1
            assert isinstance(pe.tasks[0], LinearTask)
            assert pe.tasks[0].tile_rows == 2
            assert pe.tasks[0].tile_cols == 4
            assert pe.tasks[0].fragment_slot == i
            assert pe.tasks[0].fragment_offset == i * 2
            assert len(pe.initial_sram) == 2  # weight + bias

        # Collect PE has ConcatCollectTask entries
        collect_pe = next(p for p in program.pe_programs if p.coord == (0, 3))
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
