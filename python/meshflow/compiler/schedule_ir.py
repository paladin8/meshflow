"""Schedule IR — per-PE task lists with concrete routes."""

from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


@dataclass
class TaskEntry:
    kind: str
    trigger_slot: int
    input_slot: int
    route_dest: tuple[int, int] | None = None
    route_hops: list[Direction] | None = None


@dataclass
class PESchedule:
    coord: tuple[int, int]
    tasks: list[TaskEntry]


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
