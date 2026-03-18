# Milestone 2: Compiler Skeleton and Artifact Format

## Objective

Build the Python compiler pipeline and a binary artifact format. The compiler takes a programmatic graph description, lowers it through three IR stages, and emits a MessagePack artifact. The Rust runtime gains the ability to load and execute that artifact directly from bytes.

Operators supported: `ForwardActivation` and `CollectOutput` only (same as M1). No real compute. M3 adds `Linear`.

Note: Torch model import is **not in M2**. It will be needed in M4/M5 when targeting real MLP models — at that point a `meshflow.models.torch_import` module will trace a `torch.nn.Module` and convert it to `GraphIR`. The compiler only ever sees `GraphIR` regardless of input source.

## Exit criteria

1. A `GraphIR` with forward/collect nodes compiles through all 3 passes without error.
2. The artifact serializes to MessagePack and deserializes back to an equal `RuntimeProgram`.
3. Rust deserializes a Python-generated artifact and configures a working `Simulator`.
4. `run_program(bytes, inputs)` executes end-to-end: compile -> serialize -> load -> simulate -> results.
5. A 3-node forward chain produces correct passthrough output via the artifact path.
6. Results from the artifact path match results from equivalent M1 manual setup.
7. `cargo test -p mesh_runtime` and `uv run pytest tests/python` both pass.
8. Artifact inspect tool dumps readable JSON.

---

## 1. Compiler pipeline overview

The compiler flow has three IR stages plus a final lowering to the artifact:

```
GraphIR ──► SpatialIR ──► ScheduleIR ──► RuntimeProgram (msgpack bytes)
  │             │              │
  nodes +    + coords       + routes        Rust loads bytes
  edges      (placement)    + timing        directly into Simulator
```

Each pass is a standalone function:

```
compile(graph, config)
    │
    ├── 1. place(graph, config)    -> SpatialIR
    ├── 2. route(spatial, config)  -> ScheduleIR
    └── 3. lower(schedule)         -> RuntimeProgram -> bytes
```

### 1.1 Compiler config

```python
class PlacementStrategy(Enum):
    SEQUENTIAL = "sequential"

class RoutingStrategy(Enum):
    DIMENSION_ORDERED_XY = "xy"

@dataclass
class CompilerConfig:
    placement: PlacementStrategy = PlacementStrategy.SEQUENTIAL
    routing: RoutingStrategy = RoutingStrategy.DIMENSION_ORDERED_XY
    mesh_width: int | None = None    # None = auto-size from graph
    mesh_height: int | None = None
```

Strategy enums start with one variant each. Adding a new strategy means adding an enum variant and a new function — no existing code changes. The lowering step has no strategy flag; it is a mechanical 1:1 translation.

---

## 2. IR data structures

### 2.1 Graph IR

Pure topology — nodes and edges, no placement information.

```python
class OpType(Enum):
    FORWARD = "forward"      # receives + re-emits data
    COLLECT = "collect"      # receives + stores as output

@dataclass
class Node:
    id: str
    op: OpType

@dataclass
class Edge:
    src_node: str
    src_slot: int
    dst_node: str
    dst_slot: int

@dataclass
class GraphIR:
    nodes: list[Node]
    edges: list[Edge]
```

Edges represent data dependencies: source node's output slot -> destination node's input slot. Nodes with no incoming edges are automatically identified as **input entry points** by the compiler.

### 2.2 Spatial IR

Graph IR plus placement — each node is assigned a PE coordinate.

```python
@dataclass
class PlacedNode:
    id: str
    op: OpType
    coord: tuple[int, int]

@dataclass
class PlacedEdge:
    src_node: str
    src_slot: int
    dst_node: str
    dst_slot: int

@dataclass
class SpatialIR:
    width: int
    height: int
    nodes: list[PlacedNode]
    edges: list[PlacedEdge]
```

The placement pass decides mesh dimensions (auto-sized or from config) and assigns coordinates.

### 2.3 Schedule IR

Flattened to per-PE task lists with concrete routes. This is the final pre-artifact representation.

```python
class Direction(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

@dataclass
class TaskEntry:
    kind: str                               # "forward_activation" | "collect_output"
    trigger_slot: int
    input_slot: int
    route_dest: tuple[int, int] | None      # only for forward_activation
    route_hops: list[Direction] | None       # only for forward_activation

@dataclass
class PESchedule:
    coord: tuple[int, int]
    tasks: list[TaskEntry]

@dataclass
class InputSlot:
    name: str                   # node ID from GraphIR
    coord: tuple[int, int]
    payload_slot: int

@dataclass
class ScheduleIR:
    width: int
    height: int
    pe_schedules: list[PESchedule]
    input_slots: list[InputSlot]
```

Routes are **compiler-generated** and stored in the schedule. The Rust runtime executes whatever routes it is given — it does not re-generate them. This mirrors real spatial architectures (e.g., Cerebras WSE) where the compiler programs the fabric routing at compile time.

---

## 3. Compiler passes

### 3.1 Placement pass

```python
def place(graph: GraphIR, config: CompilerConfig) -> SpatialIR
```

Dispatches on `config.placement`:

- **SEQUENTIAL**: walk nodes in topological order, assign coordinates in row-major order: `(0,0)`, `(1,0)`, `(2,0)`, etc. If `mesh_width`/`mesh_height` are `None`, auto-size the mesh to `Nx1` (single row) where N is the number of nodes. This is the simplest deterministic layout for M2; future strategies can produce 2D layouts.

Validates:
- All edge references point to existing nodes
- No duplicate node IDs
- Graph is a DAG (no cycles)

### 3.2 Routing pass

```python
def route(spatial: SpatialIR, config: CompilerConfig) -> ScheduleIR
```

Dispatches on `config.routing`:

- **DIMENSION_ORDERED_XY**: for each edge, generate a hop list from the source node's coord to the destination node's coord using X-first-then-Y ordering. The Python route generator mirrors the Rust `generate_route_xy` logic from M1.

Transforms:
- Nodes -> per-PE task entries (OpType.FORWARD -> ForwardActivation task, OpType.COLLECT -> CollectOutput task)
- Edges -> hop lists stored on tasks
- Identifies input nodes (no incoming edges) -> input slots

#### 3.2.1 Slot assignment convention

In M2, all tasks use `trigger_slot = 0` and `input_slot = 0`. This matches M1's convention where `input_slot == trigger_slot` (a task reads from the same slot that triggers it). Forwarded messages are also delivered to `payload_slot = 0` at the destination PE (matching M1's hardcoded convention in `process_execute`). This convention will be revisited in later milestones if operators need multiple input/output slots.

### 3.3 Lowering pass

```python
def lower(schedule: ScheduleIR) -> RuntimeProgram
```

No config flag — mechanical 1:1 translation from ScheduleIR to the artifact schema. Maps:
- `PESchedule` -> `PEProgram` (with empty `initial_sram` in M2)
- `TaskEntry` -> `TaskProgram` (Direction enums are converted to lowercase strings: `Direction.EAST` -> `"east"`)
- `InputSlot` -> `InputSlotProgram`
- Mesh dimensions -> `MeshProgramConfig`

---

## 4. Artifact format

### 4.1 RuntimeProgram schema

The artifact the compiler emits and the Rust runtime loads:

```python
@dataclass
class RuntimeProgram:
    version: int                            # artifact schema version, starts at 1
    mesh_config: MeshProgramConfig
    pe_programs: list[PEProgram]
    input_slots: list[InputSlotProgram]

@dataclass
class MeshProgramConfig:
    width: int
    height: int
    hop_latency: int = 1
    task_base_latency: int = 1
    max_events: int = 100_000

@dataclass
class PEProgram:
    coord: tuple[int, int]
    tasks: list[TaskProgram]
    initial_sram: dict[int, list[float]]    # slot_id -> data (empty in M2)

@dataclass
class TaskProgram:
    kind: str                               # "forward_activation" | "collect_output"
    trigger_slot: int
    input_slot: int
    route_dest: tuple[int, int] | None      # for forward_activation (debugging/inspection)
    route_hops: list[str] | None            # ["east", "north", ...] — compiler-generated

@dataclass
class InputSlotProgram:
    name: str                               # node ID from GraphIR
    coord: tuple[int, int]
    payload_slot: int
```

### 4.2 Serialization

- **Python**: `RuntimeProgram` -> MessagePack bytes via `msgpack`. Serialized as a named-field map (dict-style), not positional arrays, so the Rust deserializer can match fields by name.
- **Rust**: MessagePack bytes -> mirror serde structs via `rmp-serde`. The Rust structs use `#[derive(Deserialize)]` with default `serde` field naming (snake_case), matching the Python dict keys.
- File extension convention: `.mpk`

### 4.2.1 String-to-enum mapping conventions

All enum values are serialized as **lowercase strings** in the artifact:

| Python enum | Artifact string | Rust mapping |
|---|---|---|
| `OpType.FORWARD` / `TaskEntry.kind` | `"forward_activation"` | `TaskKind::ForwardActivation` |
| `OpType.COLLECT` / `TaskEntry.kind` | `"collect_output"` | `TaskKind::CollectOutput` |
| `Direction.NORTH` | `"north"` | `Direction::North` |
| `Direction.SOUTH` | `"south"` | `Direction::South` |
| `Direction.EAST` | `"east"` | `Direction::East` |
| `Direction.WEST` | `"west"` | `Direction::West` |

The Rust loader rejects any string not in this table with a `ProgramError`.

### 4.3 Payload separation

The artifact describes the program structure but **not the input data**. Payloads are provided at run time:

```python
result = run_program(
    program_bytes=artifact_bytes,
    inputs={"a": [1.0, 2.0, 3.0]}    # keyed by input node name
)
```

The runtime resolves each input name to the corresponding PE coordinate and slot from the artifact's `input_slots`, injects a same-PE message with the provided payload, then runs the simulation.

This separates the compiled program from its data — the same artifact can run on different inputs without recompilation.

### 4.4 Inspection utility

A CLI tool to dump an artifact as pretty-printed JSON:

```bash
uv run python -m meshflow.tools.inspect_artifact artifacts/compiled/my_program.mpk
```

Uses `msgpack` to deserialize and `orjson` (already a dependency) for JSON output.

---

## 5. Rust-side artifact loading

### 5.1 New module: `program.rs`

```
crates/mesh_runtime/src/
├── ... (existing M1 modules)
└── program.rs          # serde structs, load_program(), ProgramError
```

**Serde structs** mirror the Python artifact schema:

```rust
#[derive(Deserialize)]
struct RuntimeProgram {
    version: u32,
    mesh_config: MeshProgramConfig,
    pe_programs: Vec<PEProgram>,
    input_slots: Vec<InputSlotProgram>,
}

#[derive(Deserialize)]
struct MeshProgramConfig {
    width: u32,
    height: u32,
    hop_latency: u64,
    task_base_latency: u64,
    max_events: u64,
}

#[derive(Deserialize)]
struct PEProgram {
    coord: (u32, u32),
    tasks: Vec<TaskProgram>,
    initial_sram: HashMap<u32, Vec<f32>>,
}

#[derive(Deserialize)]
struct TaskProgram {
    kind: String,
    trigger_slot: u32,
    input_slot: u32,
    route_dest: Option<(u32, u32)>,
    route_hops: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct InputSlotProgram {
    name: String,
    coord: (u32, u32),
    payload_slot: u32,
}
```

### 5.2 Loading function

```rust
pub fn load_program(bytes: &[u8]) -> Result<LoadedProgram, ProgramError>
```

`LoadedProgram` holds the deserialized and validated program, ready to create a `Simulator`:

1. Deserialize MessagePack bytes into `RuntimeProgram`
2. Validate: coordinates in bounds, hop directions valid, input slot names unique
3. Build `SimConfig` from `mesh_config`
4. Convert `TaskProgram` entries to `TaskConfig` structs (hops from artifact, not re-generated)
5. Store `initial_sram` data (empty in M2, used for weights in M3)
6. Store `input_slots` mapping for runtime payload injection

```rust
pub fn run_with_inputs(
    &self,
    inputs: HashMap<String, Vec<f32>>,
) -> Result<SimResult, ProgramError>
```

This method on `LoadedProgram`:
1. Creates a `Simulator` from the stored config
2. Configures tasks on PEs
3. Pre-loads `initial_sram` slots
4. Resolves input names to coordinates and injects messages with provided payloads
5. Calls `sim.run()` and returns `SimResult`

### 5.3 PyO3 bridge addition

```python
# New function exposed to Python
def run_program(program_bytes: bytes, inputs: dict[str, list[float]]) -> SimResult
```

Returns the same `SimResult` pyclass from M1 (with `outputs`, `total_hops`, `total_messages`, `pe_stats`, etc.). One call: bytes + input data in, results out. The existing `run_simulation` API from M1 stays for manual/test use.

---

## 6. End-to-end flow

```python
from meshflow.compiler import compile, CompilerConfig
from meshflow.compiler.graph_ir import GraphIR, Node, Edge, OpType
from meshflow._mesh_runtime import run_program

# 1. Build a graph
graph = GraphIR(
    nodes=[
        Node(id="a", op=OpType.FORWARD),
        Node(id="b", op=OpType.FORWARD),
        Node(id="c", op=OpType.COLLECT),
    ],
    edges=[
        Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0),
        Edge(src_node="b", src_slot=0, dst_node="c", dst_slot=0),
    ],
)

# 2. Compile to artifact
artifact_bytes = compile(graph, CompilerConfig())

# 3. Execute on the Rust runtime
result = run_program(artifact_bytes, inputs={"a": [1.0, 2.0, 3.0]})
assert result.outputs[(2, 0)] == [1.0, 2.0, 3.0]  # passthrough
```

---

## 7. New dependencies

**Python:**
- `msgpack` — artifact serialization (add to `pyproject.toml` dependencies)

**Rust (add to `Cargo.toml`):**
- `rmp-serde` — MessagePack deserialization (`serde` with `derive` feature is already present from M1)

No other new dependencies. `orjson` (already present) is used by the inspect tool.

---

## 8. File layout

```
Python (new):
  python/meshflow/compiler/
    __init__.py                 # re-exports compile()
    config.py                   # CompilerConfig, PlacementStrategy, RoutingStrategy
    graph_ir.py                 # GraphIR, Node, Edge, OpType
    spatial_ir.py               # SpatialIR, PlacedNode, PlacedEdge
    schedule_ir.py              # ScheduleIR, PESchedule, TaskEntry, InputSlot, Direction
    artifact.py                 # RuntimeProgram, serialize(), deserialize()
    passes/
      __init__.py
      place.py                  # place(GraphIR, config) -> SpatialIR
      route.py                  # route(SpatialIR, config) -> ScheduleIR
      lower.py                  # lower(ScheduleIR) -> RuntimeProgram

  python/meshflow/tools/
    inspect_artifact.py         # CLI: dump .mpk as pretty JSON

  tests/python/
    test_graph_ir.py            # GraphIR construction + validation
    test_compiler_passes.py     # place, route, lower — unit tests per pass
    test_artifact.py            # serialize/deserialize round-trip
    test_end_to_end.py          # compile -> load -> run -> verify output

Rust (new):
  crates/mesh_runtime/src/
    program.rs                  # serde structs, LoadedProgram, load_program()

Rust (modified):
  crates/mesh_runtime/src/
    lib.rs                      # add mod program
    bridge.rs                   # add run_program() PyO3 function
  crates/mesh_runtime/Cargo.toml  # add rmp-serde

Python (modified):
  python/meshflow/_mesh_runtime.pyi   # add run_program() type stub
  pyproject.toml                      # add msgpack dependency
```

---

## 9. Test plan

### 9.1 Python unit tests

**`test_graph_ir.py`:**
- Construct valid graph with forward/collect nodes
- Validate edges reference existing nodes (reject invalid)
- Detect cycles (reject non-DAG)
- Identify input nodes (no incoming edges)
- Reject duplicate node IDs

**`test_compiler_passes.py`:**
- Placement: sequential assigns expected coords for N nodes
- Placement: auto-sizing produces correct mesh dimensions
- Placement: explicit mesh dimensions are respected
- Routing: XY routing generates correct hop lists for placed nodes
- Routing: same-PE routing produces empty hop list
- Routing: input slots are identified from graph topology
- Lowering: schedule IR lowers to RuntimeProgram with all fields populated
- Lowering: empty initial_sram in M2

**`test_artifact.py`:**
- Round-trip: `RuntimeProgram` -> msgpack bytes -> deserialize -> equal
- Inspect tool: artifact dumps as valid JSON
- Reject malformed bytes gracefully

### 9.2 Rust unit tests

**`program.rs` tests:**
- Deserialize valid artifact bytes
- Reject malformed MessagePack
- Reject out-of-bounds coordinates
- Reject invalid direction strings in hop lists
- Reject unknown task kind strings
- Load and run: artifact produces correct SimResult

### 9.3 End-to-end integration tests

**`test_end_to_end.py`:**
- 3-node forward chain: compile -> run_program -> verify passthrough output
- Single node collect: compile -> run_program -> verify direct output
- Artifact path matches M1 manual path: build the equivalent M1 `SimInput` manually with the same topology, run both `run_simulation` and `run_program`, compare `outputs`, `total_hops`, and `total_messages`
- Multiple input nodes: verify each receives its payload correctly

---

## 10. What is explicitly not in M2

- No real compute operators (no Linear, no matrix ops)
- No torch model import
- No weight pre-loading (initial_sram schema exists but is empty)
- No trace event export
- No SRAM capacity enforcement
- No alternative placement or routing strategies (one of each)
- No API server changes
- No streaming or incremental compilation
