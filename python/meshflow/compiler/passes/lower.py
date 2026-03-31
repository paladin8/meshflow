"""Lowering pass — mechanical translation from ScheduleIR to RuntimeProgram."""

from meshflow.compiler.artifact import (
    AddTask,
    BroadcastRouteTask,
    CollectOutputTask,
    ConcatAddTask,
    ConcatCollectForwardTask,
    ConcatCollectTask,
    ForwardActivationTask,
    InputSlotProgram,
    LinearTask,
    MatMulTask,
    MeshProgramConfig,
    PEProgram,
    RmsNormFusedTask,
    RouteTableEntry,
    RuntimeProgram,
    SoftmaxTask,
    TaskProgram,
)
from meshflow.compiler.config import CompilerConfig
from meshflow.compiler.schedule_ir import (
    AddEntry,
    BroadcastRoute,
    CollectOutputEntry,
    ConcatCollectEntry,
    ConcatCollectForwardEntry,
    ForwardActivationEntry,
    LinearEntry,
    MatMulEntry,
    ConcatAddEntry,
    RmsNormFusedEntry,
    ScheduleIR,
    SoftmaxEntry,
    TaskEntry,
)


def _lower_route(route: BroadcastRoute) -> BroadcastRouteTask:
    """Convert a schedule-IR BroadcastRoute to artifact BroadcastRouteTask.

    Strips hops and deliver_at -- routing is in per-PE routing tables now.
    """
    return BroadcastRouteTask(
        dest=route.dest,
        payload_slot=route.payload_slot,
        color=route.color,
    )


def _lower_task(task: TaskEntry) -> TaskProgram:
    """Convert a ScheduleIR task entry to the corresponding artifact task."""
    if isinstance(task, ForwardActivationEntry):
        return ForwardActivationTask(
            trigger_slot=task.trigger_slot,
            input_slot=task.input_slot,
            routes=[_lower_route(r) for r in task.routes],
        )
    if isinstance(task, CollectOutputEntry):
        return CollectOutputTask(
            trigger_slot=task.trigger_slot,
            input_slot=task.input_slot,
        )
    if isinstance(task, LinearEntry):
        return LinearTask(
            trigger_slot=task.trigger_slot,
            input_slot=task.input_slot,
            weight_slot=task.weight_slot,
            bias_slot=task.bias_slot,
            tile_rows=task.tile_rows,
            tile_cols=task.tile_cols,
            routes=[_lower_route(r) for r in task.routes],
            fragment_offset=task.fragment_offset,
        )
    if isinstance(task, ConcatCollectEntry):
        return ConcatCollectTask(
            trigger_slot=task.trigger_slot,
            num_fragments=task.num_fragments,
            total_rows=task.total_rows,
            fragment_offset=task.fragment_offset,
            fragment_rows=task.fragment_rows,
            num_positions=task.num_positions,
        )
    if isinstance(task, ConcatCollectForwardEntry):
        return ConcatCollectForwardTask(
            trigger_slot=task.trigger_slot,
            num_fragments=task.num_fragments,
            total_rows=task.total_rows,
            fragment_offset=task.fragment_offset,
            fragment_rows=task.fragment_rows,
            num_positions=task.num_positions,
            scatter=task.scatter,
            activation=task.activation,
            routes=[_lower_route(r) for r in task.routes],
        )
    if isinstance(task, AddEntry):
        return AddTask(
            trigger_slot=task.trigger_slot,
            input_slot_a=task.input_slot_a,
            input_slot_b=task.input_slot_b,
            output_slot=task.output_slot,
            routes=[_lower_route(r) for r in task.routes],
        )
    if isinstance(task, SoftmaxEntry):
        return SoftmaxTask(
            trigger_slot=task.trigger_slot,
            input_slot=task.input_slot,
            output_slot=task.output_slot,
        )
    if isinstance(task, MatMulEntry):
        return MatMulTask(
            trigger_slot=task.trigger_slot,
            matrix_slot=task.matrix_slot,
            vector_slot=task.vector_slot,
            rows=task.rows,
            cols=task.cols,
            transpose=task.transpose,
            output_slot=task.output_slot,
            routes=[_lower_route(r) for r in task.routes],
        )
    if isinstance(task, ConcatAddEntry):
        return ConcatAddTask(
            trigger_slot=task.trigger_slot,
            num_fragments=task.num_fragments,
            total_rows=task.total_rows,
            fragment_offset=task.fragment_offset,
            fragment_rows=task.fragment_rows,
            num_positions=task.num_positions,
            residual_slot=task.residual_slot,
            routes=[_lower_route(r) for r in task.routes],
        )
    if isinstance(task, RmsNormFusedEntry):
        return RmsNormFusedTask(
            trigger_slot=task.trigger_slot,
            input_slot=task.input_slot,
            gamma_slot=task.gamma_slot,
            feature_count=task.feature_count,
            eps=task.eps,
            routes=[_lower_route(r) for r in task.routes],
        )
    raise ValueError(f"unknown task entry type: {type(task)!r}")


def lower(schedule: ScheduleIR, config: CompilerConfig | None = None) -> RuntimeProgram:
    """Lower a ScheduleIR into a RuntimeProgram artifact."""
    if config is None:
        config = CompilerConfig()

    mesh_config = MeshProgramConfig(
        width=schedule.width,
        height=schedule.height,
    )

    pe_programs = [
        PEProgram(
            coord=pe.coord,
            tasks=[_lower_task(task) for task in pe.tasks],
            initial_sram=pe.initial_sram,
            sram_capacity_bytes=config.sram_capacity_bytes,
            routing_table={
                color: RouteTableEntry(
                    direction=entry.direction.value,
                    deliver_slot=entry.deliver_slot,
                )
                for color, entry in pe.routing_table.items()
            },
        )
        for pe in schedule.pe_schedules
    ]

    input_slots = [
        InputSlotProgram(
            name=slot.name,
            coord=slot.coord,
            payload_slot=slot.payload_slot,
        )
        for slot in schedule.input_slots
    ]

    return RuntimeProgram(
        version=1,
        mesh_config=mesh_config,
        pe_programs=pe_programs,
        input_slots=input_slots,
    )
