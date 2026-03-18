"""Lowering pass — mechanical translation from ScheduleIR to RuntimeProgram."""

from meshflow.compiler.artifact import (
    InputSlotProgram,
    MeshProgramConfig,
    PEProgram,
    RuntimeProgram,
    TaskProgram,
)
from meshflow.compiler.schedule_ir import ScheduleIR


def lower(schedule: ScheduleIR) -> RuntimeProgram:
    """Lower a ScheduleIR into a RuntimeProgram artifact."""
    mesh_config = MeshProgramConfig(
        width=schedule.width,
        height=schedule.height,
    )

    pe_programs = [
        PEProgram(
            coord=pe.coord,
            tasks=[
                TaskProgram(
                    kind=task.kind,
                    trigger_slot=task.trigger_slot,
                    input_slot=task.input_slot,
                    route_dest=task.route_dest,
                    route_hops=[d.value for d in task.route_hops]
                    if task.route_hops is not None
                    else None,
                )
                for task in pe.tasks
            ],
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
