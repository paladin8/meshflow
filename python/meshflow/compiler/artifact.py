"""RuntimeProgram artifact — the compiled output the Rust runtime loads."""

from dataclasses import asdict, dataclass, field
from typing import Any

import msgpack


@dataclass
class MeshProgramConfig:
    width: int
    height: int
    task_base_latency: int = 1
    cost_per_element: int = 1
    max_events: int = 100_000


@dataclass
class BroadcastRouteTask:
    """A single route in a broadcast fan-out (serialization form).

    dest: (x, y) coordinate of the final destination.
    payload_slot: SRAM slot to deliver into on the destination PE.
    color: route color ID for link multiplexing.
    """

    dest: tuple[int, int] = (0, 0)
    payload_slot: int = 0
    color: int = 0


@dataclass
class RouteTableEntry:
    """Per-PE routing table entry for the artifact.

    direction: "north", "south", "east", "west"
    deliver_slot: if set, deliver payload to this SRAM slot before forwarding.
    """

    direction: str
    deliver_slot: int | None = None


# ---------------------------------------------------------------------------
# Per-kind task dataclasses — each serializes as a flat dict with `kind`
# as the discriminator.
# ---------------------------------------------------------------------------


@dataclass
class ForwardActivationTask:
    kind: str = field(default="forward_activation", init=False)
    trigger_slot: int = 0
    input_slot: int = 0
    routes: list[BroadcastRouteTask] = field(default_factory=list)


@dataclass
class CollectOutputTask:
    kind: str = field(default="collect_output", init=False)
    trigger_slot: int = 0
    input_slot: int = 0


@dataclass
class LinearTask:
    kind: str = field(default="linear", init=False)
    trigger_slot: int = 0
    input_slot: int = 0
    weight_slot: int = 1
    bias_slot: int = 2
    tile_rows: int = 0
    tile_cols: int = 0
    routes: list[BroadcastRouteTask] = field(default_factory=list)
    fragment_offset: int = 0


@dataclass
class ConcatCollectTask:
    kind: str = field(default="concat_collect", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    total_rows: int = 0
    fragment_offset: int = 0
    fragment_rows: int = 0
    num_positions: int = 0


@dataclass
class ConcatCollectForwardTask:
    kind: str = field(default="concat_collect_forward", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    total_rows: int = 0
    fragment_offset: int = 0
    fragment_rows: int = 0
    num_positions: int = 0
    scatter: bool = False
    activation: str | None = None
    routes: list[BroadcastRouteTask] = field(default_factory=list)


@dataclass
class AddTask:
    kind: str = field(default="add", init=False)
    trigger_slot: int = 0
    input_slot_a: int = 0
    input_slot_b: int = 1
    output_slot: int = 2
    routes: list[BroadcastRouteTask] = field(default_factory=list)


@dataclass
class SoftmaxTask:
    kind: str = field(default="softmax", init=False)
    trigger_slot: int = 0
    input_slot: int = 0
    output_slot: int = 1


@dataclass
class MatMulTask:
    kind: str = field(default="mat_mul", init=False)
    trigger_slot: int = 0
    matrix_slot: int = 0
    vector_slot: int = 0
    rows: int = 0
    cols: int = 0
    transpose: bool = False
    output_slot: int = 0
    routes: list[BroadcastRouteTask] = field(default_factory=list)


@dataclass
class RmsNormPartialSumTask:
    kind: str = field(default="rms_norm_partial_sum", init=False)
    trigger_slot: int = 0
    input_slot: int = 0
    routes: list[BroadcastRouteTask] = field(default_factory=list)
    slice_offset: int = 0
    slice_size: int = 0
    feature_count: int = 0


@dataclass
class RmsNormNormalizeTask:
    kind: str = field(default="rms_norm_normalize", init=False)
    trigger_slot: int = 1
    input_slot: int = 0
    scale_slot: int = 1
    gamma_slot: int = 2
    routes: list[BroadcastRouteTask] = field(default_factory=list)
    slice_offset: int = 0
    slice_size: int = 0


@dataclass
class RmsNormReduceTask:
    kind: str = field(default="rms_norm_reduce", init=False)
    trigger_slot: int = 0
    num_tiles: int = 0
    feature_count: int = 0
    eps: float = 1e-6
    routes: list[BroadcastRouteTask] = field(default_factory=list)


TaskProgram = (
    ForwardActivationTask
    | CollectOutputTask
    | LinearTask
    | ConcatCollectTask
    | ConcatCollectForwardTask
    | AddTask
    | SoftmaxTask
    | MatMulTask
    | RmsNormPartialSumTask
    | RmsNormNormalizeTask
    | RmsNormReduceTask
)
"""Union of all task types that can appear in a PEProgram."""


@dataclass
class PEProgram:
    coord: tuple[int, int]
    tasks: list[TaskProgram]
    initial_sram: dict[int, list[float]] = field(default_factory=dict)
    sram_capacity_bytes: int | None = None
    routing_table: dict[int, RouteTableEntry] = field(default_factory=dict)


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
    result = msgpack.packb(data, use_bin_type=True)
    assert result is not None
    return result


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
                "tasks": [_task_to_dict(task) for task in pe.tasks],
                "initial_sram": {k: v for k, v in pe.initial_sram.items()},
                "sram_capacity_bytes": pe.sram_capacity_bytes,
                "routing_table": {
                    str(color): {
                        "direction": entry.direction,
                        "deliver_slot": entry.deliver_slot,
                    }
                    for color, entry in pe.routing_table.items()
                },
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


def _task_to_dict(task: TaskProgram) -> dict[str, Any]:
    """Convert a per-kind task dataclass to a flat dict."""
    d = asdict(task)
    # Convert routes list of BroadcastRouteTask dicts — dest tuple to list
    if "routes" in d:
        for route in d["routes"]:
            route["dest"] = list(route["dest"])
    return d


def _reconstruct_routes(raw_routes: list[dict[str, Any]]) -> list[BroadcastRouteTask]:
    """Reconstruct BroadcastRouteTask objects from deserialized dicts."""
    return [
        BroadcastRouteTask(
            dest=tuple(r["dest"]),
            payload_slot=r["payload_slot"],
            color=r.get("color", 0),
        )
        for r in raw_routes
    ]


def _dict_to_task(d: dict[str, Any]) -> TaskProgram:
    """Dispatch on `kind` and construct the right task dataclass."""
    kind = d["kind"]
    fields = {k: v for k, v in d.items() if k != "kind"}
    if kind == "forward_activation":
        if "routes" in fields:
            fields["routes"] = _reconstruct_routes(fields["routes"])
        return ForwardActivationTask(**fields)
    if kind == "collect_output":
        return CollectOutputTask(**fields)
    if kind == "linear":
        if "routes" in fields:
            fields["routes"] = _reconstruct_routes(fields["routes"])
        return LinearTask(**fields)
    if kind == "concat_collect":
        return ConcatCollectTask(**fields)
    if kind == "concat_collect_forward":
        if "routes" in fields:
            fields["routes"] = _reconstruct_routes(fields["routes"])
        return ConcatCollectForwardTask(**fields)
    if kind == "add":
        if "routes" in fields:
            fields["routes"] = _reconstruct_routes(fields["routes"])
        return AddTask(**fields)
    if kind == "softmax":
        return SoftmaxTask(**fields)
    if kind == "mat_mul":
        if "routes" in fields:
            fields["routes"] = _reconstruct_routes(fields["routes"])
        return MatMulTask(**fields)
    if kind == "rms_norm_partial_sum":
        if "routes" in fields:
            fields["routes"] = _reconstruct_routes(fields["routes"])
        return RmsNormPartialSumTask(**fields)
    if kind == "rms_norm_normalize":
        if "routes" in fields:
            fields["routes"] = _reconstruct_routes(fields["routes"])
        return RmsNormNormalizeTask(**fields)
    if kind == "rms_norm_reduce":
        if "routes" in fields:
            fields["routes"] = _reconstruct_routes(fields["routes"])
        return RmsNormReduceTask(**fields)
    raise ValueError(f"unknown task kind: {kind!r}")


def _dict_to_program(raw: dict[str, Any]) -> RuntimeProgram:
    """Convert a deserialized dict back into a RuntimeProgram."""
    # Filter out removed fields for backward compatibility with pre-M11 artifacts
    known_fields = {f.name for f in MeshProgramConfig.__dataclass_fields__.values()}
    raw_config = {k: v for k, v in raw["mesh_config"].items() if k in known_fields}
    mesh_config = MeshProgramConfig(**raw_config)

    pe_programs = [
        PEProgram(
            coord=tuple(pe["coord"]),  # type: ignore[arg-type]
            tasks=[_dict_to_task(task) for task in pe["tasks"]],
            initial_sram={int(k): v for k, v in pe["initial_sram"].items()},
            sram_capacity_bytes=pe.get("sram_capacity_bytes"),
            routing_table={
                int(color): RouteTableEntry(
                    direction=entry["direction"],
                    deliver_slot=entry.get("deliver_slot"),
                )
                for color, entry in pe.get("routing_table", {}).items()
            },
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
