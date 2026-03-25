"""Lowering pass — mechanical translation from ScheduleIR to RuntimeProgram."""

from meshflow.compiler.artifact import (
    AddTask,
    CollectOutputTask,
    ConcatCollectForwardTask,
    ConcatCollectTask,
    ForwardActivationTask,
    InputSlotProgram,
    LinearTask,
    MatMulTask,
    MeshProgramConfig,
    PEProgram,
    RmsNormNormalizeTask,
    RmsNormPartialSumTask,
    RmsNormReduceTask,
    RuntimeProgram,
    SoftmaxTask,
    TaskProgram,
)
from meshflow.compiler.config import CompilerConfig
from meshflow.compiler.schedule_ir import (
    AddEntry,
    CollectOutputEntry,
    ConcatCollectEntry,
    ConcatCollectForwardEntry,
    ForwardActivationEntry,
    LinearEntry,
    MatMulEntry,
    RmsNormNormalizeEntry,
    RmsNormPartialSumEntry,
    RmsNormReduceEntry,
    ScheduleIR,
    SoftmaxEntry,
    TaskEntry,
)


def _lower_task(task: TaskEntry) -> TaskProgram:
    """Convert a ScheduleIR task entry to the corresponding artifact task."""
    if isinstance(task, ForwardActivationEntry):
        return ForwardActivationTask(
            trigger_slot=task.trigger_slot,
            input_slot=task.input_slot,
            route_dest=task.route_dest,
            route_hops=[d.value for d in task.route_hops],
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
            route_dest=task.route_dest,
            route_hops=[d.value for d in task.route_hops],
            fragment_slot=task.fragment_slot,
            fragment_offset=task.fragment_offset,
        )
    if isinstance(task, ConcatCollectEntry):
        return ConcatCollectTask(
            trigger_slot=task.trigger_slot,
            num_fragments=task.num_fragments,
            total_rows=task.total_rows,
            fragment_offset=task.fragment_offset,
            num_positions=task.num_positions,
        )
    if isinstance(task, ConcatCollectForwardEntry):
        return ConcatCollectForwardTask(
            trigger_slot=task.trigger_slot,
            num_fragments=task.num_fragments,
            total_rows=task.total_rows,
            fragment_offset=task.fragment_offset,
            num_positions=task.num_positions,
            scatter=task.scatter,
            activation=task.activation,
            route_dests=[(coord, [d.value for d in hops]) for coord, hops in task.route_dests],
        )
    if isinstance(task, AddEntry):
        return AddTask(
            trigger_slot=task.trigger_slot,
            input_slot_a=task.input_slot_a,
            input_slot_b=task.input_slot_b,
            output_slot=task.output_slot,
            output_dests=[(coord, [d.value for d in hops]) for coord, hops in task.output_dests],
            payload_slots=list(task.payload_slots),
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
            output_dests=[(coord, [d.value for d in hops]) for coord, hops in task.output_dests],
            payload_slots=list(task.payload_slots),
        )
    if isinstance(task, RmsNormPartialSumEntry):
        return RmsNormPartialSumTask(
            trigger_slot=task.trigger_slot,
            input_slot=task.input_slot,
            reduce_dest=task.reduce_dest,
            reduce_hops=[d.value for d in task.reduce_hops],
            partial_sum_slot=task.partial_sum_slot,
            slice_offset=task.slice_offset,
            slice_size=task.slice_size,
        )
    if isinstance(task, RmsNormNormalizeEntry):
        return RmsNormNormalizeTask(
            trigger_slot=task.trigger_slot,
            input_slot=task.input_slot,
            scale_slot=task.scale_slot,
            gamma_slot=task.gamma_slot,
            output_dests=[(coord, [d.value for d in hops]) for coord, hops in task.output_dests],
            payload_slots=list(task.payload_slots),
            slice_offset=task.slice_offset,
            slice_size=task.slice_size,
        )
    if isinstance(task, RmsNormReduceEntry):
        return RmsNormReduceTask(
            trigger_slot=task.trigger_slot,
            num_tiles=task.num_tiles,
            feature_count=task.feature_count,
            eps=task.eps,
            tile_dests=[(coord, [d.value for d in hops]) for coord, hops in task.tile_dests],
            scale_slot=task.scale_slot,
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
