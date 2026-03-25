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
    fragment_offset: int = 0


@dataclass
class ConcatCollectEntry:
    kind: str = field(default="concat_collect", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    total_rows: int = 0
    fragment_offset: int = 0
    num_positions: int = 1


@dataclass
class ConcatCollectForwardEntry:
    kind: str = field(default="concat_collect_forward", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    total_rows: int = 0
    fragment_offset: int = 0
    num_positions: int = 1
    scatter: bool = False
    activation: str | None = None
    route_dests: list[tuple[tuple[int, int], list[Direction]]] = field(default_factory=list)


@dataclass
class AddEntry:
    kind: str = field(default="add", init=False)
    trigger_slot: int = 0
    input_slot_a: int = 0
    input_slot_b: int = 1
    output_slot: int = 2
    output_dests: list[tuple[tuple[int, int], list[Direction]]] = field(default_factory=list)
    payload_slots: list[int] = field(default_factory=list)


@dataclass
class SoftmaxEntry:
    kind: str = field(default="softmax", init=False)
    trigger_slot: int = 0
    input_slot: int = 0
    output_slot: int = 1


@dataclass
class MatMulEntry:
    kind: str = field(default="mat_mul", init=False)
    trigger_slot: int = 0
    operand_slots: list[int] = field(default_factory=list)
    num_dynamic_operands: int = 0
    output_slot: int = 0
    output_dests: list[tuple[tuple[int, int], list[Direction]]] = field(default_factory=list)
    payload_slots: list[int] = field(default_factory=list)


@dataclass
class RmsNormPartialSumEntry:
    kind: str = field(default="rms_norm_partial_sum", init=False)
    trigger_slot: int = 0
    input_slot: int = 0
    reduce_dest: tuple[int, int] = (0, 0)
    reduce_hops: list[Direction] = field(default_factory=list)
    partial_sum_slot: int = 0
    slice_offset: int = 0
    slice_size: int = 0


@dataclass
class RmsNormNormalizeEntry:
    kind: str = field(default="rms_norm_normalize", init=False)
    trigger_slot: int = 1
    input_slot: int = 0
    scale_slot: int = 1
    gamma_slot: int = 2
    output_dests: list[tuple[tuple[int, int], list[Direction]]] = field(default_factory=list)
    payload_slots: list[int] = field(default_factory=list)
    slice_offset: int = 0
    slice_size: int = 0


@dataclass
class RmsNormReduceEntry:
    kind: str = field(default="rms_norm_reduce", init=False)
    trigger_slot: int = 0
    num_tiles: int = 0
    feature_count: int = 0
    eps: float = 1e-6
    tile_dests: list[tuple[tuple[int, int], list[Direction]]] = field(default_factory=list)
    scale_slot: int = 1


TaskEntry = (
    ForwardActivationEntry
    | CollectOutputEntry
    | LinearEntry
    | ConcatCollectEntry
    | ConcatCollectForwardEntry
    | AddEntry
    | SoftmaxEntry
    | MatMulEntry
    | RmsNormPartialSumEntry
    | RmsNormNormalizeEntry
    | RmsNormReduceEntry
)


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
