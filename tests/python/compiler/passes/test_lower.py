"""Tests for the lowering pass."""

from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.artifact import (
    AddTask,
    ForwardActivationTask,
    MatMulTask,
    RmsNormNormalizeTask,
    RmsNormPartialSumTask,
    RmsNormReduceTask,
    SoftmaxTask,
)
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.compiler.passes.expand import expand
from meshflow.compiler.passes.lower import _lower_task, lower
from meshflow.compiler.passes.place import place
from meshflow.compiler.passes.route import route
from meshflow.compiler.schedule_ir import (
    AddEntry,
    Direction,
    MatMulEntry,
    RmsNormNormalizeEntry,
    RmsNormPartialSumEntry,
    RmsNormReduceEntry,
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
        assert task.route_hops == ["east"]

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

        assert program.mesh_config.hop_latency == 1
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
            output_dests=[((2, 0), [Direction.EAST])],
            payload_slots=[0, 1],
        )
        task = _lower_task(entry)
        assert isinstance(task, AddTask)
        assert task.kind == "add"
        assert task.trigger_slot == 1
        assert task.input_slot_a == 0
        assert task.input_slot_b == 1
        assert task.output_slot == 2
        assert task.output_dests == [((2, 0), ["east"])]
        assert task.payload_slots == [0, 1]

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
            trigger_slot=4,
            operand_slots=[0, 1, 2, 3, 4],
            num_dynamic_operands=4,
            output_slot=9,
            output_dests=[((1, 0), [Direction.EAST])],
            payload_slots=[],
        )
        task = _lower_task(entry)
        assert isinstance(task, MatMulTask)
        assert task.kind == "mat_mul"
        assert task.trigger_slot == 4
        assert task.operand_slots == [0, 1, 2, 3, 4]
        assert task.num_dynamic_operands == 4
        assert task.output_slot == 9
        assert task.output_dests == [((1, 0), ["east"])]

    def test_lower_rms_norm_partial_sum_entry(self) -> None:
        entry = RmsNormPartialSumEntry(
            trigger_slot=0,
            input_slot=0,
            reduce_dest=(0, 4),
            reduce_hops=[Direction.NORTH, Direction.NORTH],
            partial_sum_slot=2,
        )
        task = _lower_task(entry)
        assert isinstance(task, RmsNormPartialSumTask)
        assert task.kind == "rms_norm_partial_sum"
        assert task.trigger_slot == 0
        assert task.input_slot == 0
        assert task.reduce_dest == (0, 4)
        assert task.reduce_hops == ["north", "north"]
        assert task.partial_sum_slot == 2

    def test_lower_rms_norm_normalize_entry(self) -> None:
        entry = RmsNormNormalizeEntry(
            trigger_slot=1,
            input_slot=0,
            scale_slot=1,
            gamma_slot=2,
            output_dests=[((1, 0), [Direction.EAST])],
            payload_slots=[0, 1, 2],
        )
        task = _lower_task(entry)
        assert isinstance(task, RmsNormNormalizeTask)
        assert task.kind == "rms_norm_normalize"
        assert task.trigger_slot == 1
        assert task.input_slot == 0
        assert task.scale_slot == 1
        assert task.gamma_slot == 2
        assert task.output_dests == [((1, 0), ["east"])]
        assert task.payload_slots == [0, 1, 2]

    def test_lower_rms_norm_reduce_entry(self) -> None:
        entry = RmsNormReduceEntry(
            trigger_slot=0,
            num_tiles=3,
            feature_count=8,
            eps=1e-6,
            tile_dests=[
                ((0, 0), [Direction.SOUTH, Direction.SOUTH]),
                ((0, 1), [Direction.SOUTH]),
            ],
            scale_slot=1,
        )
        task = _lower_task(entry)
        assert isinstance(task, RmsNormReduceTask)
        assert task.kind == "rms_norm_reduce"
        assert task.trigger_slot == 0
        assert task.num_tiles == 3
        assert task.feature_count == 8
        assert abs(task.eps - 1e-6) < 1e-10
        assert task.tile_dests == [
            ((0, 0), ["south", "south"]),
            ((0, 1), ["south"]),
        ]
        assert task.scale_slot == 1

    def test_lower_rmsnorm_full_pipeline(self) -> None:
        """End-to-end: FORWARD → RMSNORM → COLLECT through all passes."""
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
        # Check that the program has PE programs for tile PEs, reduce PE, etc.
        assert len(program.pe_programs) > 0

        # Find a tile PE and check its tasks
        tile_pe = next(
            pe
            for pe in program.pe_programs
            if any(isinstance(t, RmsNormPartialSumTask) for t in pe.tasks)
        )
        ps_tasks = [t for t in tile_pe.tasks if isinstance(t, RmsNormPartialSumTask)]
        assert len(ps_tasks) == 1
        assert ps_tasks[0].reduce_hops  # should have at least one hop direction

        norm_tasks = [t for t in tile_pe.tasks if isinstance(t, RmsNormNormalizeTask)]
        assert len(norm_tasks) == 1

        # Find the reduce PE
        reduce_pe = next(
            pe
            for pe in program.pe_programs
            if any(isinstance(t, RmsNormReduceTask) for t in pe.tasks)
        )
        reduce_tasks = [t for t in reduce_pe.tasks if isinstance(t, RmsNormReduceTask)]
        assert len(reduce_tasks) == 3  # one per partial sum slot

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
        # seq_len=2: 2 QK^T entries + 3 AV entries = 5 MatMul tasks
        assert len(matmul_tasks) == 5
        softmax_tasks = [t for t in attn_pe.tasks if isinstance(t, SoftmaxTask)]
        assert len(softmax_tasks) == 1

        # Verify direction strings are lowered
        for t in matmul_tasks:
            for _, hops in t.output_dests:
                for h in hops:
                    assert h in ("north", "south", "east", "west")
