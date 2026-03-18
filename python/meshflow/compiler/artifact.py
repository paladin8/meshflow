"""RuntimeProgram artifact — the compiled output the Rust runtime loads."""

from dataclasses import asdict, dataclass, field
from typing import Any

import msgpack


@dataclass
class MeshProgramConfig:
    width: int
    height: int
    hop_latency: int = 1
    task_base_latency: int = 1
    max_events: int = 100_000


@dataclass
class TaskProgram:
    kind: str
    trigger_slot: int
    input_slot: int
    route_dest: tuple[int, int] | None = None
    route_hops: list[str] | None = None


@dataclass
class PEProgram:
    coord: tuple[int, int]
    tasks: list[TaskProgram]
    initial_sram: dict[int, list[float]] = field(default_factory=dict)


@dataclass
class InputSlotProgram:
    name: str
    coord: tuple[int, int]
    payload_slot: int


@dataclass
class RuntimeProgram:
    version: int
    mesh_config: MeshProgramConfig
    pe_programs: list[PEProgram]
    input_slots: list[InputSlotProgram]


def serialize(program: RuntimeProgram) -> bytes:
    """Serialize a RuntimeProgram to MessagePack bytes (named-field map style)."""
    data = _program_to_dict(program)
    return msgpack.packb(data, use_bin_type=True)


def deserialize(data: bytes) -> RuntimeProgram:
    """Deserialize MessagePack bytes into a RuntimeProgram."""
    try:
        raw = msgpack.unpackb(data, raw=False, strict_map_key=False)
    except Exception as e:
        raise ValueError(f"failed to unpack artifact bytes: {e}") from e
    try:
        return _dict_to_program(raw)
    except (KeyError, TypeError) as e:
        raise ValueError(f"invalid artifact structure: {e}") from e


def _program_to_dict(program: RuntimeProgram) -> dict[str, Any]:
    """Convert RuntimeProgram to a plain dict for serialization.

    Tuples are converted to lists (msgpack has no tuple type).
    Dict keys in initial_sram are converted to ints.
    """
    return {
        "version": program.version,
        "mesh_config": asdict(program.mesh_config),
        "pe_programs": [
            {
                "coord": list(pe.coord),
                "tasks": [
                    {
                        "kind": task.kind,
                        "trigger_slot": task.trigger_slot,
                        "input_slot": task.input_slot,
                        "route_dest": list(task.route_dest)
                        if task.route_dest is not None
                        else None,
                        "route_hops": task.route_hops,
                    }
                    for task in pe.tasks
                ],
                "initial_sram": {k: v for k, v in pe.initial_sram.items()},
            }
            for pe in program.pe_programs
        ],
        "input_slots": [
            {
                "name": slot.name,
                "coord": list(slot.coord),
                "payload_slot": slot.payload_slot,
            }
            for slot in program.input_slots
        ],
    }


def _dict_to_program(raw: dict[str, Any]) -> RuntimeProgram:
    """Convert a deserialized dict back into a RuntimeProgram."""
    mesh_config = MeshProgramConfig(**raw["mesh_config"])

    pe_programs = [
        PEProgram(
            coord=tuple(pe["coord"]),  # type: ignore[arg-type]
            tasks=[
                TaskProgram(
                    kind=task["kind"],
                    trigger_slot=task["trigger_slot"],
                    input_slot=task["input_slot"],
                    route_dest=tuple(task["route_dest"])
                    if task["route_dest"] is not None
                    else None,  # type: ignore[arg-type]
                    route_hops=task["route_hops"],
                )
                for task in pe["tasks"]
            ],
            initial_sram={int(k): v for k, v in pe["initial_sram"].items()},
        )
        for pe in raw["pe_programs"]
    ]

    input_slots = [
        InputSlotProgram(
            name=slot["name"],
            coord=tuple(slot["coord"]),  # type: ignore[arg-type]
            payload_slot=slot["payload_slot"],
        )
        for slot in raw["input_slots"]
    ]

    return RuntimeProgram(
        version=raw["version"],
        mesh_config=mesh_config,
        pe_programs=pe_programs,
        input_slots=input_slots,
    )
