"""Tests for the routing pass."""

import numpy as np
import pytest
from meshflow.compiler import CompilerConfig
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.compiler.passes.expand import expand
from meshflow.compiler.passes.place import place
from meshflow.compiler.passes.route import _generate_route_xy, _try_linear_broadcast, route
from meshflow.compiler.schedule_ir import (
    AddEntry,
    BroadcastRoute,
    ConcatCollectForwardEntry,
    Direction,
    MatMulEntry,
    RmsNormFusedEntry,
    SoftmaxEntry,
)


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
        assert task.routes[0].dest == (1, 0)
        assert task.routes[0].hops == [Direction.EAST]

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
        assert task.routes[0].hops == [Direction.EAST]

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
        assert task.routes[0].hops == [Direction.NORTH]

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
        assert a_pe.tasks[0].routes[0].hops == [Direction.EAST, Direction.NORTH]


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

        # Find collect PE coord dynamically
        collect_coord = next(
            p.coord
            for p in schedule.pe_schedules
            if any(t.kind == "concat_collect" for t in p.tasks)
        )

        # Tile PEs have linear tasks — find them by task kind
        tile_pes = [p for p in schedule.pe_schedules if any(t.kind == "linear" for t in p.tasks)]
        assert len(tile_pes) == 3
        for pe in tile_pes:
            assert len(pe.tasks) == 1
            task = pe.tasks[0]
            assert task.kind == "linear"
            assert task.weight_slot == 1
            assert task.bias_slot == 2
            assert task.tile_rows == 2
            assert task.tile_cols == 4
            assert task.routes[0].dest == collect_coord

        # Verify fragment indexing is correct across tiles
        fragment_slots = sorted(pe.tasks[0].routes[0].payload_slot for pe in tile_pes)
        assert fragment_slots == [0, 1, 2]
        for pe in tile_pes:
            task = pe.tasks[0]
            assert task.fragment_offset == task.routes[0].payload_slot * 2

        # Collect PE has 3 concat_collect tasks
        collect_pe = next(p for p in schedule.pe_schedules if p.coord == collect_coord)
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

        # Find tile PEs by task kind and check SRAM contents
        tile_pes = [p for p in schedule.pe_schedules if any(t.kind == "linear" for t in p.tasks)]
        assert len(tile_pes) == 3
        for pe in tile_pes:
            i = pe.tasks[0].routes[0].payload_slot
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

        # l1 collect PE: find by ConcatCollectForwardEntry tasks in column 0
        l1_collect = next(
            p
            for p in schedule.pe_schedules
            if p.coord[0] == 0 and any(isinstance(t, ConcatCollectForwardEntry) for t in p.tasks)
        )
        assert len(l1_collect.tasks) == 3
        for i, task in enumerate(l1_collect.tasks):
            assert isinstance(task, ConcatCollectForwardEntry)
            assert task.kind == "concat_collect_forward"
            assert task.trigger_slot == i
            assert task.num_fragments == 3
            assert task.total_rows == 6
            assert task.activation == "relu"
            # Broadcast detection: 3 l2 tile PEs at X=1, same payload_slot.
            # With middle-collect layout, tiles may be split above/below the
            # l2 collect PE, producing multiple broadcast routes.
            total_deliveries = sum(len(r.deliver_at) for r in task.routes)
            dest_count = len(task.routes) + total_deliveries
            assert dest_count == 3  # routes reach all 3 l2 tile PEs
            for r in task.routes:
                assert r.payload_slot == 0

    def test_terminal_collect_has_concat_tasks(self) -> None:
        graph = self._make_mlp_graph()
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, self._make_mlp_weights())

        # l2 collect PE: find by concat_collect tasks in column 1
        l2_collect = next(
            p
            for p in schedule.pe_schedules
            if p.coord[0] == 1 and any(t.kind == "concat_collect" for t in p.tasks)
        )
        assert len(l2_collect.tasks) == 3
        for task in l2_collect.tasks:
            assert task.kind == "concat_collect"

    def test_inter_layer_route_hops(self) -> None:
        """Route from l1 collect to l2 tiles: broadcast east then up/down."""
        graph = self._make_mlp_graph()
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, self._make_mlp_weights())

        l1_collect = next(
            p
            for p in schedule.pe_schedules
            if p.coord[0] == 0 and any(isinstance(t, ConcatCollectForwardEntry) for t in p.tasks)
        )
        task = l1_collect.tasks[0]
        assert isinstance(task, ConcatCollectForwardEntry)

        # Broadcast detection collapses 3 routes into broadcast(s).
        # Verify broadcasts go east to column 1 and use deliver_at for
        # intermediate tile PEs.  Exact coordinates depend on stagger offset.
        assert len(task.routes) >= 1
        for r in task.routes:
            assert Direction.EAST in r.hops
            # Dest should be in column 1
            assert r.dest[0] == 1

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

        # l1 tile PEs (column 0) with linear tasks have weights
        l1_tiles = [
            p
            for p in schedule.pe_schedules
            if p.coord[0] == 0 and any(t.kind == "linear" for t in p.tasks)
        ]
        assert len(l1_tiles) == 3
        for pe in l1_tiles:
            assert 1 in pe.initial_sram  # weight
            assert 2 in pe.initial_sram  # bias

        # l2 tile PEs (column 1) with linear tasks have weights
        l2_tiles = [
            p
            for p in schedule.pe_schedules
            if p.coord[0] == 1 and any(t.kind == "linear" for t in p.tasks)
        ]
        assert len(l2_tiles) == 3
        for pe in l2_tiles:
            assert 1 in pe.initial_sram
            assert 2 in pe.initial_sram

    def test_no_activation_on_direct_linear(self) -> None:
        """Linear -> Linear (no RELU): forward tasks have activation=None."""
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

        # l1 collect PE: find by ConcatCollectForwardEntry in column 0
        l1_collect = next(
            p
            for p in schedule.pe_schedules
            if p.coord[0] == 0 and any(isinstance(t, ConcatCollectForwardEntry) for t in p.tasks)
        )
        task = l1_collect.tasks[0]
        assert isinstance(task, ConcatCollectForwardEntry)
        assert task.activation is None


class TestRmsNormRouting:
    def _make_rmsnorm_graph(self, feature_count: int = 4) -> GraphIR:
        return GraphIR(
            nodes=[
                Node(id="fwd", op=OpType.FORWARD),
                Node(
                    id="rn",
                    op=OpType.RMSNORM,
                    attrs={"eps": 1e-6, "feature_count": feature_count},
                ),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="fwd", src_slot=0, dst_node="rn", dst_slot=0),
                Edge(src_node="rn", src_slot=0, dst_node="col", dst_slot=0),
            ],
        )

    def _make_rmsnorm_weights(self, feature_count: int = 4) -> dict[str, dict[str, np.ndarray]]:
        return {
            "rn": {
                "gamma": np.ones(feature_count, dtype=np.float32),
            }
        }

    def test_rmsnorm_fused_pe_has_fused_entry(self) -> None:
        """The fused RMSNorm PE has a single RmsNormFusedEntry task."""
        graph = self._make_rmsnorm_graph()
        config = CompilerConfig()
        weights = self._make_rmsnorm_weights()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, weights)

        # Find the fused PE
        fused_pes = [
            p
            for p in schedule.pe_schedules
            if any(isinstance(t, RmsNormFusedEntry) for t in p.tasks)
        ]
        assert len(fused_pes) == 1

        fused_pe = fused_pes[0]
        fused_tasks = [t for t in fused_pe.tasks if isinstance(t, RmsNormFusedEntry)]
        assert len(fused_tasks) == 1

        task = fused_tasks[0]
        assert task.kind == "rms_norm_fused"
        assert task.trigger_slot == 0
        assert task.input_slot == 0
        assert task.gamma_slot == 1
        assert task.feature_count == 4
        assert abs(task.eps - 1e-6) < 1e-10

    def test_rmsnorm_fused_routes_to_downstream(self) -> None:
        """The fused PE routes its output to the downstream COLLECT PE."""
        graph = self._make_rmsnorm_graph()
        config = CompilerConfig()
        weights = self._make_rmsnorm_weights()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, weights)

        fused_pe = next(
            p
            for p in schedule.pe_schedules
            if any(isinstance(t, RmsNormFusedEntry) for t in p.tasks)
        )
        fused_task = next(t for t in fused_pe.tasks if isinstance(t, RmsNormFusedEntry))
        assert len(fused_task.routes) >= 1

        # The output should reach the collect PE
        collect_coord = next(
            p.coord
            for p in schedule.pe_schedules
            if any(t.kind == "collect_output" for t in p.tasks)
        )
        dest_coords = [r.dest for r in fused_task.routes]
        assert collect_coord in dest_coords

    def test_rmsnorm_gamma_in_sram(self) -> None:
        """Gamma weights loaded on the fused PE at SRAM slot 1."""
        graph = self._make_rmsnorm_graph()
        config = CompilerConfig()
        gamma = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        weights = {"rn": {"gamma": gamma}}
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, weights)

        fused_pe = next(
            p
            for p in schedule.pe_schedules
            if any(isinstance(t, RmsNormFusedEntry) for t in p.tasks)
        )
        # Full gamma loaded into SRAM slot 1 (gamma_slot)
        assert 1 in fused_pe.initial_sram
        assert fused_pe.initial_sram[1] == pytest.approx([1.0, 2.0, 3.0, 4.0])


class TestAttentionRouting:
    def test_attention_pe_has_matmul_softmax_matmul(self) -> None:
        """QK^T MatMul → Softmax → AV MatMul chain on attention PEs."""
        graph = GraphIR(
            nodes=[
                Node(id="qkt", op=OpType.MATMUL, attrs={"seq_len": 2}),
                Node(id="sm", op=OpType.SOFTMAX),
                Node(id="av", op=OpType.MATMUL, attrs={"seq_len": 2}),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="qkt", src_slot=0, dst_node="sm", dst_slot=0),
                Edge(src_node="sm", src_slot=0, dst_node="av", dst_slot=0),
                Edge(src_node="av", src_slot=0, dst_node="col", dst_slot=0),
            ],
        )
        config = CompilerConfig()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config)

        # New slot layout: Q=0, K=1, V=2, QK^T=3, softmax=4, AV=5
        # 2 QK^T entries (trigger on Q=0 and K=1) + 1 Softmax + 2 AV entries (trigger on V=2 and softmax=4)
        # Find an attention PE by its task content (not hardcoded coordinate)
        attn_pe = next(
            p for p in schedule.pe_schedules if any(isinstance(t, MatMulEntry) for t in p.tasks)
        )
        matmul_tasks = [t for t in attn_pe.tasks if isinstance(t, MatMulEntry)]
        softmax_tasks = [t for t in attn_pe.tasks if isinstance(t, SoftmaxEntry)]

        # QK^T: trigger on Q (0) and K (1)
        qkt_tasks = [t for t in matmul_tasks if t.output_slot == 3]
        assert len(qkt_tasks) == 2
        assert {t.trigger_slot for t in qkt_tasks} == {0, 1}
        assert qkt_tasks[0].matrix_slot == 1  # K matrix
        assert qkt_tasks[0].vector_slot == 0  # Q vector
        assert qkt_tasks[0].transpose is False

        # Softmax
        assert len(softmax_tasks) == 1
        sm = softmax_tasks[0]
        assert sm.trigger_slot == 3
        assert sm.input_slot == 3
        assert sm.output_slot == 4

        # AV: trigger on V (2) and softmax output (4)
        av_tasks = [t for t in matmul_tasks if t.output_slot == 5]
        assert len(av_tasks) == 2
        assert {t.trigger_slot for t in av_tasks} == {2, 4}
        assert av_tasks[0].matrix_slot == 2  # V matrix
        assert av_tasks[0].vector_slot == 4  # softmax weights
        assert av_tasks[0].transpose is True

    def test_attention_slot_layout(self) -> None:
        """Verify SRAM slot assignments follow the spec for seq_len=4."""
        graph = GraphIR(
            nodes=[
                Node(id="qkt", op=OpType.MATMUL, attrs={"seq_len": 4}),
                Node(id="sm", op=OpType.SOFTMAX),
                Node(id="av", op=OpType.MATMUL, attrs={"seq_len": 4}),
            ],
            edges=[
                Edge(src_node="qkt", src_slot=0, dst_node="sm", dst_slot=0),
                Edge(src_node="sm", src_slot=0, dst_node="av", dst_slot=0),
            ],
        )
        config = CompilerConfig()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config)

        attn_pe = next(
            p for p in schedule.pe_schedules if any(isinstance(t, MatMulEntry) for t in p.tasks)
        )
        matmul_tasks = [t for t in attn_pe.tasks if isinstance(t, MatMulEntry)]

        # QK^T: 2 entries (trigger on Q=0 and K=1), output slot 3
        qkt_tasks = [t for t in matmul_tasks if t.output_slot == 3]
        assert len(qkt_tasks) == 2
        assert qkt_tasks[0].matrix_slot == 1  # K
        assert qkt_tasks[0].vector_slot == 0  # Q

        sm = next(t for t in attn_pe.tasks if isinstance(t, SoftmaxEntry))
        assert sm.input_slot == 3
        assert sm.output_slot == 4

        # AV: 2 entries (trigger on V=2 and softmax=4), output slot 5
        av_tasks = [t for t in matmul_tasks if t.output_slot == 5]
        assert len(av_tasks) == 2
        assert av_tasks[0].matrix_slot == 2  # V
        assert av_tasks[0].vector_slot == 4  # softmax
        assert av_tasks[0].transpose is True

    def test_attention_av_routes_to_downstream(self) -> None:
        """AV result routes to downstream PEs."""
        graph = GraphIR(
            nodes=[
                Node(id="qkt", op=OpType.MATMUL, attrs={"seq_len": 2}),
                Node(id="sm", op=OpType.SOFTMAX),
                Node(id="av", op=OpType.MATMUL, attrs={"seq_len": 2}),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="qkt", src_slot=0, dst_node="sm", dst_slot=0),
                Edge(src_node="sm", src_slot=0, dst_node="av", dst_slot=0),
                Edge(src_node="av", src_slot=0, dst_node="col", dst_slot=0),
            ],
        )
        config = CompilerConfig()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config)

        attn_pe = next(
            p for p in schedule.pe_schedules if any(isinstance(t, MatMulEntry) for t in p.tasks)
        )
        av_tasks = [t for t in attn_pe.tasks if isinstance(t, MatMulEntry) and t.output_slot == 5]
        assert len(av_tasks) > 0
        # All AV entries share the same routes
        av = av_tasks[0]
        assert len(av.routes) >= 1
        dest_coords = [r.dest for r in av.routes]
        # AV routes to attention collect PE (in same column)
        assert len(dest_coords) >= 1


class TestAddRouting:
    def test_add_creates_add_entry(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="add", op=OpType.ADD),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="add", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="add", dst_slot=1),
                Edge(src_node="add", src_slot=0, dst_node="col", dst_slot=0),
            ],
        )
        config = CompilerConfig()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config)

        add_pe = next(p for p in schedule.pe_schedules if p.coord == (2, 0))
        # Two AddEntry tasks — one per trigger slot (has_slot guard in runtime)
        add_tasks = [t for t in add_pe.tasks if isinstance(t, AddEntry)]
        assert len(add_tasks) == 2
        assert {t.trigger_slot for t in add_tasks} == {0, 1}
        for task in add_tasks:
            assert task.input_slot_a == 0
            assert task.input_slot_b == 1
            assert task.output_slot == 2

    def test_add_routes_to_downstream(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="add", op=OpType.ADD),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="add", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="add", dst_slot=1),
                Edge(src_node="add", src_slot=0, dst_node="col", dst_slot=0),
            ],
        )
        config = CompilerConfig()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config)

        add_pe = next(p for p in schedule.pe_schedules if p.coord == (2, 0))
        task = add_pe.tasks[0]
        assert isinstance(task, AddEntry)
        assert len(task.routes) >= 1
        # collect PE is at (3, 0)
        dest_coords = [r.dest for r in task.routes]
        assert (3, 0) in dest_coords


class TestSoftmaxRouting:
    def test_standalone_softmax(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="fwd", op=OpType.FORWARD),
                Node(id="sm", op=OpType.SOFTMAX),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="fwd", src_slot=0, dst_node="sm", dst_slot=0),
                Edge(src_node="sm", src_slot=0, dst_node="col", dst_slot=0),
            ],
        )
        config = CompilerConfig()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config)

        sm_pe = next(p for p in schedule.pe_schedules if p.coord == (1, 0))
        assert len(sm_pe.tasks) == 1
        task = sm_pe.tasks[0]
        assert isinstance(task, SoftmaxEntry)
        assert task.kind == "softmax"
        assert task.trigger_slot == 0
        assert task.input_slot == 0
        assert task.output_slot == 1


class TestForwardBroadcast:
    def test_forward_broadcast_multiple_destinations(self) -> None:
        """FORWARD node with multiple outgoing edges generates multiple tasks."""
        graph = GraphIR(
            nodes=[
                Node(id="fwd", op=OpType.FORWARD),
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="fwd", src_slot=0, dst_node="a", dst_slot=0),
                Edge(src_node="fwd", src_slot=0, dst_node="b", dst_slot=0),
                Edge(src_node="a", src_slot=0, dst_node="col", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="col", dst_slot=1),
            ],
        )
        config = CompilerConfig()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config)

        fwd_pe = next(p for p in schedule.pe_schedules if p.coord == (0, 0))
        # Should have 2 ForwardActivation tasks (broadcast)
        assert len(fwd_pe.tasks) == 2
        for task in fwd_pe.tasks:
            assert task.kind == "forward_activation"


class TestBroadcastDetection:
    """Unit tests for _try_linear_broadcast and route-pass broadcast detection."""

    def test_linear_broadcast_detection_column(self) -> None:
        """Multiple same-column destinations collapse into one broadcast route.

        Source at (0, 3), destinations at (1, 0), (1, 1), (1, 2) — all in column 1.
        Expected: 1 route with dest=(1, 0) (farthest south) and deliver_at=[1, 2]
        (one per intermediate PE above it, sorted closest-first).
        """
        dests = [
            BroadcastRoute(dest=(1, 0), hops=[], deliver_at=[], payload_slot=1),
            BroadcastRoute(dest=(1, 1), hops=[], deliver_at=[], payload_slot=1),
            BroadcastRoute(dest=(1, 2), hops=[], deliver_at=[], payload_slot=1),
        ]
        result = _try_linear_broadcast((0, 3), dests)

        assert len(result) == 1, f"expected 1 broadcast route, got {len(result)}: {result}"
        r = result[0]
        assert r.dest == (1, 0), f"expected dest=(1,0), got {r.dest}"
        assert len(r.deliver_at) > 0, "broadcast route should have non-empty deliver_at"
        # Intermediate PEs (1,2) and (1,1) should be in deliver_at
        assert len(r.deliver_at) == 2, f"expected 2 intermediate deliveries, got {r.deliver_at}"

    def test_single_dest_no_broadcast(self) -> None:
        """A single destination is returned as-is (no broadcast optimisation).

        _try_linear_broadcast returns the original list unchanged when there is
        only one destination, and deliver_at should be empty.
        """
        dests = [
            BroadcastRoute(dest=(1, 0), hops=[Direction.EAST], deliver_at=[], payload_slot=0),
        ]
        result = _try_linear_broadcast((0, 0), dests)

        assert len(result) == 1
        assert result[0].deliver_at == [], (
            f"single-destination route should have empty deliver_at, got {result[0].deliver_at}"
        )

    def test_non_column_fallback(self) -> None:
        """Single destinations per column are returned as point-to-point.

        When each column has only one destination, no broadcast is possible
        within any group — each route is returned with empty deliver_at.
        """
        dests = [
            BroadcastRoute(dest=(1, 0), hops=[], deliver_at=[], payload_slot=0),
            BroadcastRoute(dest=(2, 0), hops=[], deliver_at=[], payload_slot=0),
            BroadcastRoute(dest=(3, 0), hops=[], deliver_at=[], payload_slot=0),
        ]
        result = _try_linear_broadcast((0, 0), dests)

        # Each column has only 1 dest, so all remain point-to-point
        assert len(result) == 3, f"expected 3 separate routes, got {len(result)}"
        for r in result:
            assert r.deliver_at == [], (
                f"single-dest-per-column routes should have empty deliver_at, got {r.deliver_at}"
            )

    def test_bidirectional_broadcast(self) -> None:
        """Destinations on both sides of the source produce 2 broadcast routes.

        Source at (0, 2), destinations at (0, 0), (0, 1), (0, 3), (0, 4).
        Expected: 2 routes — one South (to (0,0) via (0,1)) and one North
        (to (0,4) via (0,3)).
        """
        dests = [
            BroadcastRoute(dest=(0, 0), hops=[], deliver_at=[], payload_slot=1),
            BroadcastRoute(dest=(0, 1), hops=[], deliver_at=[], payload_slot=1),
            BroadcastRoute(dest=(0, 3), hops=[], deliver_at=[], payload_slot=1),
            BroadcastRoute(dest=(0, 4), hops=[], deliver_at=[], payload_slot=1),
        ]
        result = _try_linear_broadcast((0, 2), dests)

        assert len(result) == 2, f"expected 2 broadcast routes, got {len(result)}: {result}"

        # Sort by dest Y to identify South vs North route
        south = [r for r in result if r.dest[1] < 2]
        north = [r for r in result if r.dest[1] > 2]
        assert len(south) == 1, f"expected 1 South route, got {len(south)}"
        assert len(north) == 1, f"expected 1 North route, got {len(north)}"

        # South route: dest=(0,0), delivers at (0,1) intermediate
        assert south[0].dest == (0, 0)
        assert len(south[0].deliver_at) == 1, (
            f"South route should have 1 intermediate delivery, got {south[0].deliver_at}"
        )

        # North route: dest=(0,4), delivers at (0,3) intermediate
        assert north[0].dest == (0, 4)
        assert len(north[0].deliver_at) == 1, (
            f"North route should have 1 intermediate delivery, got {north[0].deliver_at}"
        )

    def test_multi_column_broadcast_grouping(self) -> None:
        """Destinations across multiple columns produce one broadcast per column.

        Source at (0, 3), destinations in columns 1, 2, 3 with 3 tiles each.
        Expected: 3 broadcast routes (one per column), each delivering to ~3 tiles.
        """
        dests = []
        for col in [1, 2, 3]:
            for row in [0, 1, 2]:
                dests.append(
                    BroadcastRoute(dest=(col, row), hops=[], deliver_at=[], payload_slot=0)
                )
        result = _try_linear_broadcast((0, 3), dests)

        # 3 groups (one per column), each collapsed into 1 broadcast route
        assert len(result) == 3, f"expected 3 broadcast routes, got {len(result)}: {result}"
        for r in result:
            assert len(r.deliver_at) > 0, (
                f"each multi-dest column should broadcast, got deliver_at={r.deliver_at}"
            )
        # Each route should target a different column
        dest_xs = {r.dest[0] for r in result}
        assert dest_xs == {1, 2, 3}, f"expected routes to columns 1,2,3, got {dest_xs}"

    def test_multi_column_mixed_slots(self) -> None:
        """Different payload_slots in the same column produce separate groups.

        2 destinations in column 1 with slot=0, 2 destinations in column 1 with slot=1.
        Expected: 2 broadcast routes (one per payload_slot).
        """
        dests = [
            BroadcastRoute(dest=(1, 0), hops=[], deliver_at=[], payload_slot=0),
            BroadcastRoute(dest=(1, 1), hops=[], deliver_at=[], payload_slot=0),
            BroadcastRoute(dest=(1, 2), hops=[], deliver_at=[], payload_slot=1),
            BroadcastRoute(dest=(1, 3), hops=[], deliver_at=[], payload_slot=1),
        ]
        result = _try_linear_broadcast((0, 0), dests)

        # 2 groups: (x=1, slot=0) and (x=1, slot=1)
        assert len(result) == 2, f"expected 2 broadcast routes, got {len(result)}: {result}"
        slots = {r.payload_slot for r in result}
        assert slots == {0, 1}, f"expected routes with slots 0 and 1, got {slots}"
        for r in result:
            assert len(r.deliver_at) > 0, (
                f"each 2-dest group should broadcast, got deliver_at={r.deliver_at}"
            )
