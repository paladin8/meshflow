"""Lowering pass — mechanical translation from ScheduleIR to RuntimeProgram."""

from meshflow.compiler.artifact import (
    CollectOutputTask,
    ForwardActivationTask,
    InputSlotProgram,
    MeshProgramConfig,
    PEProgram,
    RuntimeProgram,
)
from meshflow.compiler.schedule_ir import ScheduleIR, TaskEntry


def _lower_task(
    task: TaskEntry,
) -> ForwardActivationTask | CollectOutputTask:
    """Convert a ScheduleIR TaskEntry to the corresponding artifact task."""
    if task.kind == "forward_activation":
        assert task.route_dest is not None
        assert task.route_hops is not None
        return ForwardActivationTask(
            trigger_slot=task.trigger_slot,
            input_slot=task.input_slot,
            route_dest=task.route_dest,
            route_hops=[d.value for d in task.route_hops],
        )
    if task.kind == "collect_output":
        return CollectOutputTask(
            trigger_slot=task.trigger_slot,
            input_slot=task.input_slot,
        )
    raise ValueError(f"unknown task kind: {task.kind!r}")


def lower(schedule: ScheduleIR) -> RuntimeProgram:
    """Lower a ScheduleIR into a RuntimeProgram artifact."""
    mesh_config = MeshProgramConfig(
        width=schedule.width,
        height=schedule.height,
    )

    pe_programs = [
        PEProgram(
            coord=pe.coord,
            tasks=[_lower_task(task) for task in pe.tasks],
            initial_sram={},
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
