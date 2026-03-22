"""Tests for the routing pass."""

import numpy as np
import pytest
from meshflow.compiler import CompilerConfig
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.compiler.passes.expand import expand
from meshflow.compiler.passes.place import place
from meshflow.compiler.passes.route import _generate_route_xy, route
from meshflow.compiler.schedule_ir import ConcatCollectForwardEntry, Direction


class TestRouting:
    def test_xy_routing_hops(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
        )
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())
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
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())
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
        expanded = expand(graph, config)
        spatial = place(expanded, config)
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
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())
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
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())
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
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())
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
        expanded = expand(graph, config)
        spatial = place(expanded, config)
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

        expanded = expand(graph, config)
        spatial = place(expanded, config)
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

        expanded = expand(graph, config)
        spatial = place(expanded, config)
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

        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, weights)

        # 3 input slots, all named "linear1" (broadcast)
        assert len(schedule.input_slots) == 3
        for slot in schedule.input_slots:
            assert slot.name == "linear1"
            assert slot.payload_slot == 0


class TestMultiLayerRouting:
    def _make_mlp_graph(self) -> GraphIR:
        """Linear(4,6) → ReLU → Linear(6,3) with mesh_height=4."""
        return GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )

    def _make_mlp_weights(self) -> dict[str, dict[str, np.ndarray]]:
        rng = np.random.default_rng(42)
        return {
            "l1": {
                "weight": rng.standard_normal((6, 4)).astype(np.float32),
                "bias": rng.standard_normal(6).astype(np.float32),
            },
            "l2": {
                "weight": rng.standard_normal((3, 6)).astype(np.float32),
                "bias": rng.standard_normal(3).astype(np.float32),
            },
        }

    def test_intermediate_collect_has_forward_tasks(self) -> None:
        graph = self._make_mlp_graph()
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, self._make_mlp_weights())

        # l1 collect at (0, 3) should have ConcatCollectForwardEntry tasks
        l1_collect = next(p for p in schedule.pe_schedules if p.coord == (0, 3))
        assert len(l1_collect.tasks) == 3
        for i, task in enumerate(l1_collect.tasks):
            assert isinstance(task, ConcatCollectForwardEntry)
            assert task.kind == "concat_collect_forward"
            assert task.trigger_slot == i
            assert task.num_fragments == 3
            assert task.total_rows == 6
            assert task.activation == "relu"
            # Routes to l2 tile PEs at (1, 0), (1, 1), (1, 2)
            assert len(task.route_dests) == 3
            dest_coords = [coord for coord, _ in task.route_dests]
            assert dest_coords == [(1, 0), (1, 1), (1, 2)]

    def test_terminal_collect_has_concat_tasks(self) -> None:
        graph = self._make_mlp_graph()
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, self._make_mlp_weights())

        # l2 collect at (1, 3) should have ConcatCollectEntry tasks (terminal)
        l2_collect = next(p for p in schedule.pe_schedules if p.coord == (1, 3))
        assert len(l2_collect.tasks) == 3
        for task in l2_collect.tasks:
            assert task.kind == "concat_collect"

    def test_inter_layer_route_hops(self) -> None:
        """Route from l1 collect (0,3) to l2 tiles: east then south."""
        graph = self._make_mlp_graph()
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, self._make_mlp_weights())

        l1_collect = next(p for p in schedule.pe_schedules if p.coord == (0, 3))
        task = l1_collect.tasks[0]
        assert isinstance(task, ConcatCollectForwardEntry)

        # (0,3) → (1,0): 1 east, 3 south
        coord_0, hops_0 = task.route_dests[0]
        assert coord_0 == (1, 0)
        assert hops_0 == [Direction.EAST, Direction.SOUTH, Direction.SOUTH, Direction.SOUTH]

        # (0,3) → (1,1): 1 east, 2 south
        coord_1, hops_1 = task.route_dests[1]
        assert coord_1 == (1, 1)
        assert hops_1 == [Direction.EAST, Direction.SOUTH, Direction.SOUTH]

        # (0,3) → (1,2): 1 east, 1 south
        coord_2, hops_2 = task.route_dests[2]
        assert coord_2 == (1, 2)
        assert hops_2 == [Direction.EAST, Direction.SOUTH]

    def test_only_first_layer_has_input_slots(self) -> None:
        graph = self._make_mlp_graph()
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, self._make_mlp_weights())

        # Only l1 tile PEs are external inputs
        assert len(schedule.input_slots) == 3
        for slot in schedule.input_slots:
            assert slot.name == "l1"
            assert slot.coord[0] == 0  # column 0

    def test_both_layers_have_weights_in_sram(self) -> None:
        graph = self._make_mlp_graph()
        config = CompilerConfig(mesh_height=4)
        weights = self._make_mlp_weights()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, weights)

        # l1 tiles at (0, 0)-(0, 2) have weights
        for i in range(3):
            pe = next(p for p in schedule.pe_schedules if p.coord == (0, i))
            assert 1 in pe.initial_sram  # weight
            assert 2 in pe.initial_sram  # bias

        # l2 tiles at (1, 0)-(1, 2) have weights
        for i in range(3):
            pe = next(p for p in schedule.pe_schedules if p.coord == (1, i))
            assert 1 in pe.initial_sram
            assert 2 in pe.initial_sram

    def test_no_activation_on_direct_linear(self) -> None:
        """Linear → Linear (no RELU): forward tasks have activation=None."""
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6}),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[Edge(src_node="l1", src_slot=0, dst_node="l2", dst_slot=0)],
        )
        rng = np.random.default_rng(42)
        weights = {
            "l1": {
                "weight": rng.standard_normal((6, 4)).astype(np.float32),
                "bias": rng.standard_normal(6).astype(np.float32),
            },
            "l2": {
                "weight": rng.standard_normal((3, 6)).astype(np.float32),
                "bias": rng.standard_normal(3).astype(np.float32),
            },
        }
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, weights)

        l1_collect = next(p for p in schedule.pe_schedules if p.coord == (0, 3))
        task = l1_collect.tasks[0]
        assert isinstance(task, ConcatCollectForwardEntry)
        assert task.activation is None
