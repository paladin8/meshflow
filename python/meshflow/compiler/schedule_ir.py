"""Schedule IR — per-PE task lists with concrete routes."""

from dataclasses import dataclass, field
from enum import Enum


class Direction(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


@dataclass
class RouteTableEntry:
    """Per-PE routing table entry: forward a message in a direction.

    direction: which way to forward the message.
    deliver_slot: if set, deliver payload to this SRAM slot before forwarding
                  (DeliverAndForward semantics). None means forward-only.
    """

    direction: Direction
    deliver_slot: int | None = None


@dataclass
class BroadcastRoute:
    """A single route in a broadcast fan-out.

    dest: (x, y) coordinate of the final destination.
    hops: ordered Direction list from source to destination.
    deliver_at: hop indices for intermediate delivery (empty = point-to-point).
    payload_slot: SRAM slot to deliver into on the destination PE.
    """

    dest: tuple[int, int] = (0, 0)
    hops: list[Direction] = field(default_factory=list)
    deliver_at: list[int] = field(default_factory=list)
    payload_slot: int = 0
    color: int = 0


# ---------------------------------------------------------------------------
# Per-kind task entries
# ---------------------------------------------------------------------------


@dataclass
class ForwardActivationEntry:
    kind: str = field(default="forward_activation", init=False)
    trigger_slot: int = 0
    input_slot: int = 0
    routes: list[BroadcastRoute] = field(default_factory=list)


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
    routes: list[BroadcastRoute] = field(default_factory=list)
    fragment_offset: int = 0


@dataclass
class ConcatCollectEntry:
    kind: str = field(default="concat_collect", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    total_rows: int = 0
    fragment_offset: int = 0
    fragment_rows: int = 0
    num_positions: int = 0


@dataclass
class ConcatCollectForwardEntry:
    kind: str = field(default="concat_collect_forward", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    total_rows: int = 0
    fragment_offset: int = 0
    fragment_rows: int = 0
    num_positions: int = 0
    scatter: bool = False
    activation: str | None = None
    routes: list[BroadcastRoute] = field(default_factory=list)


@dataclass
class AddEntry:
    kind: str = field(default="add", init=False)
    trigger_slot: int = 0
    input_slot_a: int = 0
    input_slot_b: int = 1
    output_slot: int = 2
    routes: list[BroadcastRoute] = field(default_factory=list)


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
    matrix_slot: int = 0
    vector_slot: int = 0
    rows: int = 0
    cols: int = 0
    transpose: bool = False
    output_slot: int = 0
    routes: list[BroadcastRoute] = field(default_factory=list)


@dataclass
class RmsNormPartialSumEntry:
    kind: str = field(default="rms_norm_partial_sum", init=False)
    trigger_slot: int = 0
    input_slot: int = 0
    routes: list[BroadcastRoute] = field(default_factory=list)
    slice_offset: int = 0
    slice_size: int = 0
    feature_count: int = 0


@dataclass
class RmsNormNormalizeEntry:
    kind: str = field(default="rms_norm_normalize", init=False)
    trigger_slot: int = 1
    input_slot: int = 0
    scale_slot: int = 1
    gamma_slot: int = 2
    routes: list[BroadcastRoute] = field(default_factory=list)
    slice_offset: int = 0
    slice_size: int = 0


@dataclass
class RmsNormReduceEntry:
    kind: str = field(default="rms_norm_reduce", init=False)
    trigger_slot: int = 0
    num_tiles: int = 0
    feature_count: int = 0
    eps: float = 1e-6
    routes: list[BroadcastRoute] = field(default_factory=list)


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
    routing_table: dict[int, RouteTableEntry] = field(default_factory=dict)


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
