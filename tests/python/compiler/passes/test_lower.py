"""Tests for the lowering pass."""

from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.artifact import (
    AddTask,
    BroadcastRouteTask,
    ForwardActivationTask,
    MatMulTask,
    RmsNormFusedTask,
    SoftmaxTask,
)
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.compiler.passes.expand import expand
from meshflow.compiler.passes.lower import _lower_task, lower
from meshflow.compiler.passes.place import place
from meshflow.compiler.passes.route import route
from meshflow.compiler.schedule_ir import (
    AddEntry,
    BroadcastRoute,
    Direction,
    MatMulEntry,
    SoftmaxEntry,
)


class TestLowering:
    def test_lower_simple_chain(self) -> None:
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
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())
        schedule = route(spatial, CompilerConfig())
        program = lower(schedule)

        a_prog = next(pe for pe in program.pe_programs if pe.coord == (0, 0))
        task = a_prog.tasks[0]
        assert isinstance(task, ForwardActivationTask)
        assert len(task.routes) == 1
        assert task.routes[0].dest == (1, 0)

    def test_lower_empty_initial_sram(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="a", op=OpType.COLLECT)],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())
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

        assert program.mesh_config.task_base_latency == 1
        assert program.mesh_config.max_events == 100_000

    def test_lower_sram_capacity_from_config(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="a", op=OpType.COLLECT)],
            edges=[],
        )
        config = CompilerConfig(sram_capacity_bytes=32768)
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config)
        program = lower(schedule, config)

        for pe in program.pe_programs:
            assert pe.sram_capacity_bytes == 32768

    def test_lower_default_sram_capacity(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="a", op=OpType.COLLECT)],
            edges=[],
        )
        program = compile(graph)

        for pe in program.pe_programs:
            assert pe.sram_capacity_bytes == 65536


class TestLowerNewTasks:
    """Lowering tests for the 6 new task types."""

    def test_lower_add_entry(self) -> None:
        entry = AddEntry(
            trigger_slot=1,
            input_slot_a=0,
            input_slot_b=1,
            output_slot=2,
            routes=[
                BroadcastRoute(dest=(2, 0), hops=[Direction.EAST], payload_slot=0),
                BroadcastRoute(dest=(2, 1), hops=[Direction.EAST, Direction.NORTH], payload_slot=1),
            ],
        )
        task = _lower_task(entry)
        assert isinstance(task, AddTask)
        assert task.kind == "add"
        assert task.trigger_slot == 1
        assert task.input_slot_a == 0
        assert task.input_slot_b == 1
        assert task.output_slot == 2
        assert len(task.routes) == 2
        assert task.routes[0] == BroadcastRouteTask(dest=(2, 0), payload_slot=0)
        assert task.routes[1] == BroadcastRouteTask(dest=(2, 1), payload_slot=1)

    def test_lower_softmax_entry(self) -> None:
        entry = SoftmaxEntry(trigger_slot=5, input_slot=5, output_slot=6)
        task = _lower_task(entry)
        assert isinstance(task, SoftmaxTask)
        assert task.kind == "softmax"
        assert task.trigger_slot == 5
        assert task.input_slot == 5
        assert task.output_slot == 6

    def test_lower_mat_mul_entry(self) -> None:
        entry = MatMulEntry(
            trigger_slot=0,
            matrix_slot=1,
            vector_slot=0,
            rows=3,
            cols=2,
            transpose=False,
            output_slot=3,
            routes=[BroadcastRoute(dest=(1, 0), hops=[Direction.EAST], payload_slot=0)],
        )
        task = _lower_task(entry)
        assert isinstance(task, MatMulTask)
        assert task.kind == "mat_mul"
        assert task.matrix_slot == 1
        assert task.vector_slot == 0
        assert task.rows == 3
        assert task.cols == 2
        assert task.transpose is False
        assert task.output_slot == 3
        assert len(task.routes) == 1
        assert task.routes[0] == BroadcastRouteTask(dest=(1, 0), payload_slot=0)

    def test_lower_rmsnorm_full_pipeline(self) -> None:
        """End-to-end: FORWARD → RMSNORM → COLLECT through all passes (fused architecture)."""
        import numpy as np

        graph = GraphIR(
            nodes=[
                Node(id="fwd", op=OpType.FORWARD),
                Node(id="rn", op=OpType.RMSNORM, attrs={"eps": 1e-6, "feature_count": 3}),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="fwd", src_slot=0, dst_node="rn", dst_slot=0),
                Edge(src_node="rn", src_slot=0, dst_node="col", dst_slot=0),
            ],
        )
        config = CompilerConfig()
        weights = {"rn": {"gamma": np.ones(3, dtype=np.float32)}}
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, weights)
        program = lower(schedule, config)

        assert program.version == 1
        assert len(program.pe_programs) > 0

        # Find the fused RMSNorm PE and check its task
        fused_pe = next(
            pe
            for pe in program.pe_programs
            if any(isinstance(t, RmsNormFusedTask) for t in pe.tasks)
        )
        fused_tasks = [t for t in fused_pe.tasks if isinstance(t, RmsNormFusedTask)]
        assert len(fused_tasks) == 1
        assert fused_tasks[0].kind == "rms_norm_fused"
        assert fused_tasks[0].feature_count == 3
        assert abs(fused_tasks[0].eps - 1e-6) < 1e-10
        assert fused_tasks[0].input_slot == 0
        assert fused_tasks[0].gamma_slot == 1
        assert len(fused_tasks[0].routes) >= 1  # routes to downstream collect

        # Gamma weights should be loaded in SRAM slot 1
        assert 1 in fused_pe.initial_sram
        assert fused_pe.initial_sram[1] == [1.0, 1.0, 1.0]

    def test_lower_add_full_pipeline(self) -> None:
        """End-to-end: FORWARD + FORWARD → ADD → COLLECT."""
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
        program = lower(schedule, config)

        # Find the ADD PE
        add_pe = next(
            pe for pe in program.pe_programs if any(isinstance(t, AddTask) for t in pe.tasks)
        )
        add_tasks = [t for t in add_pe.tasks if isinstance(t, AddTask)]
        # Two entries: one per trigger slot (has_slot guard in runtime)
        assert len(add_tasks) == 2
        assert {t.trigger_slot for t in add_tasks} == {0, 1}
        assert add_tasks[0].input_slot_a == 0
        assert add_tasks[0].input_slot_b == 1

    def test_lower_attention_full_pipeline(self) -> None:
        """End-to-end: QK^T → Softmax → AV → COLLECT."""
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
        program = lower(schedule, config)

        # Find attention PE
        attn_pe = next(
            pe for pe in program.pe_programs if any(isinstance(t, MatMulTask) for t in pe.tasks)
        )
        matmul_tasks = [t for t in attn_pe.tasks if isinstance(t, MatMulTask)]
        # 2 QK^T entries (trigger on Q=0, K=1) + 2 AV entries (trigger on V=2, softmax=4)
        assert len(matmul_tasks) == 4
        softmax_tasks = [t for t in attn_pe.tasks if isinstance(t, SoftmaxTask)]
        assert len(softmax_tasks) == 1

        # Verify routes are present (hops stripped in Phase 3 lowering)
        for t in matmul_tasks:
            for r in t.routes:
                assert isinstance(r, BroadcastRouteTask)
