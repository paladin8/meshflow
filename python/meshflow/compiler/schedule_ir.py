"""Schedule IR — per-PE task lists with concrete routes."""

from dataclasses import dataclass, field
from enum import Enum


class Direction(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


# ---------------------------------------------------------------------------
# Per-kind task entries
# ---------------------------------------------------------------------------


@dataclass
class ForwardActivationEntry:
    kind: str = field(default="forward_activation", init=False)
    trigger_slot: int = 0
    input_slot: int = 0
    route_dest: tuple[int, int] = (0, 0)
    route_hops: list[Direction] = field(default_factory=list)


@dataclass
class CollectOutputEntry:
    kind: str = field(default="collect_output", init=False)
    trigger_slot: int = 0
    input_slot: int = 0


@dataclass
class LinearEntry:
    kind: str = field(default="linear", init=False)
    trigger_slot: int = 0
    input_slot: int = 0
    weight_slot: int = 1
    bias_slot: int = 2
    tile_rows: int = 0
    tile_cols: int = 0
    route_dest: tuple[int, int] = (0, 0)
    route_hops: list[Direction] = field(default_factory=list)
    fragment_slot: int = 0


@dataclass
class ConcatCollectEntry:
    kind: str = field(default="concat_collect", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    rows_per_fragment: int = 0


TaskEntry = ForwardActivationEntry | CollectOutputEntry | LinearEntry | ConcatCollectEntry


@dataclass
class PESchedule:
    coord: tuple[int, int]
    tasks: list[TaskEntry]
    initial_sram: dict[int, list[float]] = field(default_factory=dict)


@dataclass
class InputSlot:
    name: str
    coord: tuple[int, int]
    payload_slot: int


@dataclass
class ScheduleIR:
    width: int
    height: int
    pe_schedules: list[PESchedule]
    input_slots: list[InputSlot]
