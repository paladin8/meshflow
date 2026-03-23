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


# ---------------------------------------------------------------------------
# Per-kind task dataclasses — each serializes as a flat dict with `kind`
# as the discriminator.
# ---------------------------------------------------------------------------


@dataclass
class ForwardActivationTask:
    kind: str = field(default="forward_activation", init=False)
    trigger_slot: int = 0
    input_slot: int = 0
    route_dest: tuple[int, int] = (0, 0)
    route_hops: list[str] = field(default_factory=list)


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
    route_dest: tuple[int, int] = (0, 0)
    route_hops: list[str] = field(default_factory=list)
    fragment_slot: int = 0
    fragment_offset: int = 0


@dataclass
class ConcatCollectTask:
    kind: str = field(default="concat_collect", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    total_rows: int = 0
    fragment_offset: int = 0


@dataclass
class ConcatCollectForwardTask:
    kind: str = field(default="concat_collect_forward", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    total_rows: int = 0
    fragment_offset: int = 0
    activation: str | None = None
    route_dests: list[tuple[tuple[int, int], list[str]]] = field(default_factory=list)


TaskProgram = (
    ForwardActivationTask
    | CollectOutputTask
    | LinearTask
    | ConcatCollectTask
    | ConcatCollectForwardTask
)
"""Union of all task types that can appear in a PEProgram."""


@dataclass
class PEProgram:
    coord: tuple[int, int]
    tasks: list[TaskProgram]
    initial_sram: dict[int, list[float]] = field(default_factory=dict)
    sram_capacity_bytes: int | None = None


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
    # Convert tuples to lists for msgpack
    if "route_dest" in d and d["route_dest"] is not None:
        d["route_dest"] = list(d["route_dest"])
    # Convert route_dests list of (coord_tuple, hops) for ConcatCollectForwardTask
    if "route_dests" in d:
        d["route_dests"] = [[list(coord), hops] for coord, hops in d["route_dests"]]
    return d


def _dict_to_task(d: dict[str, Any]) -> TaskProgram:
    """Dispatch on `kind` and construct the right task dataclass."""
    kind = d["kind"]
    fields = {k: v for k, v in d.items() if k != "kind"}
    if kind == "forward_activation":
        if fields.get("route_dest") is not None:
            fields["route_dest"] = tuple(fields["route_dest"])
        return ForwardActivationTask(**fields)
    if kind == "collect_output":
        return CollectOutputTask(**fields)
    if kind == "linear":
        if fields.get("route_dest") is not None:
            fields["route_dest"] = tuple(fields["route_dest"])
        return LinearTask(**fields)
    if kind == "concat_collect":
        return ConcatCollectTask(**fields)
    if kind == "concat_collect_forward":
        # Convert route_dests: [[coord_list, hops], ...] → [(coord_tuple, hops), ...]
        if "route_dests" in fields:
            fields["route_dests"] = [(tuple(coord), hops) for coord, hops in fields["route_dests"]]
        return ConcatCollectForwardTask(**fields)
    raise ValueError(f"unknown task kind: {kind!r}")


def _dict_to_program(raw: dict[str, Any]) -> RuntimeProgram:
    """Convert a deserialized dict back into a RuntimeProgram."""
    mesh_config = MeshProgramConfig(**raw["mesh_config"])

    pe_programs = [
        PEProgram(
            coord=tuple(pe["coord"]),  # type: ignore[arg-type]
            tasks=[_dict_to_task(task) for task in pe["tasks"]],
            initial_sram={int(k): v for k, v in pe["initial_sram"].items()},
            sram_capacity_bytes=pe.get("sram_capacity_bytes"),
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
