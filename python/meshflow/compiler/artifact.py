"""RuntimeProgram artifact — the compiled output the Rust runtime loads."""

from dataclasses import dataclass, field


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
