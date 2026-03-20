# Milestone 3: Single Linear Operator Execution

## Objective

Execute a single linear layer (`y = Wx + b`) distributed across multiple PEs via row-wise tiling. The compiler tiles weights, embeds them in the artifact's `initial_sram`, broadcasts the input activation to all tile PEs, and gathers output fragments at a collect node. Correctness is verified against `torch.nn.functional.linear`.

This is the first milestone with real compute — PEs do matrix-vector multiplication, not just data forwarding.

## Exit criteria

1. A single `LINEAR` node compiles, serializes, loads, and runs end-to-end.
2. Simulator output matches `torch.nn.functional.linear` within `atol=1e-6`.
3. Tiling works: multiple PEs each compute a fragment, collect node has the full result.
4. Existing M2 tests still pass (forward/collect passthrough unchanged).
5. All linters clean (`cargo fmt`, `clippy`, `ruff`, `mypy`).
6. Rust unit tests cover `Linear` task compute with known values.

---

## 1. Data flow overview

For a linear layer with `out_features=6, in_features=4, num_tiles=3`:

- Mesh is `4x1`: 3 tile PEs at `(0,0)`, `(1,0)`, `(2,0)` + 1 collect PE at `(3,0)`.
- Compiler tiles `W (6,4)` row-wise into 3 chunks of `(2,4)` and `b (6,)` into 3 chunks of `(2,)`.
- Weight and bias tiles are pre-loaded into each PE's SRAM via `initial_sram`.
- Input `x (4,)` is broadcast — same payload sent as a message to all 3 tile PEs.
- Each tile PE computes `y_i = W_i @ x + b_i`, producing a `(2,)` output fragment.
- Each tile PE routes its fragment to the collect PE at `(3,0)`.
- Collect PE concatenates fragments in order → full `y (6,)`.

### 1.1 SRAM slot convention (tile PEs)

| Slot | Contents | Loaded by |
|------|----------|-----------|
| 0 | Input activation | Message delivery (runtime) |
| 1 | Weight tile (row-major) | `initial_sram` (artifact) |
| 2 | Bias tile | `initial_sram` (artifact) |

### 1.2 Fragment collection mechanism

Each tile PE routes its output fragment to the collect PE, but each sends to a **distinct payload slot** based on tile index: tile 0 sends to slot 0, tile 1 to slot 1, tile 2 to slot 2. This avoids SRAM overwrites on the collect PE.

The collect PE uses a new `ConcatCollect` task kind (not the existing `CollectOutput`). It is configured with `num_fragments` and `rows_per_fragment`. The compiler registers N `ConcatCollect` `TaskConfig` entries on the collect PE — one per fragment slot (`trigger_slot=0`, `trigger_slot=1`, ..., `trigger_slot=N-1`). This works with the existing trigger model where `triggered_tasks()` matches `trigger_slot == written_slot`.

**Accumulator model:** Instead of storing each fragment in a separate SRAM slot (which would require unbounded slots), the collect PE uses a single pre-allocated output buffer of size `num_fragments * rows_per_fragment`. Each fragment delivery triggers a `ConcatCollect` task that writes the fragment data into the buffer at offset `trigger_slot * rows_per_fragment`. A counter tracks arrivals; once counter == `num_fragments`, the completed buffer moves to `outputs`. This uses O(1) SRAM slots regardless of tile count, which is realistic for limited-SRAM hardware.

---

## 2. Python compiler changes

### 2.1 GraphIR additions

`OpType` gains a new variant:

```python
class OpType(Enum):
    FORWARD = "forward"
    COLLECT = "collect"
    LINEAR = "linear"       # new
```

`Node` gains an optional `attrs` field for operator metadata:

```python
@dataclass
class Node:
    id: str
    op: OpType
    attrs: dict[str, Any] | None = None   # new
```

Linear nodes require `attrs={"in_features": int, "out_features": int}`. Validation enforces this.

### 2.2 Compile API change

`compile()` gains a `weights` parameter:

```python
def compile(
    graph: GraphIR,
    config: CompilerConfig | None = None,
    weights: dict[str, dict[str, np.ndarray]] | None = None,
) -> RuntimeProgram:
```

`weights` maps node IDs to named tensors. For linear nodes:

```python
weights={"linear1": {"weight": W, "bias": b}}
```

Where `W` is `np.ndarray` of shape `(out_features, in_features)` and `b` is `(out_features,)`.

`compile()` validates that every `LINEAR` node has a corresponding entry in `weights` with the correct shapes.

### 2.3 Placement pass

When the placer encounters a `LINEAR` node, it expands it into multiple placed nodes:

- The node becomes N tile placements + 1 collect placement.
- N is determined by mesh dimensions: if `mesh_width` is set, `N = mesh_width - 1` (reserving one column for collect). If `mesh_width` is `None`, `N = out_features` (one row per PE, maximally parallel).
- `rows_per_pe = out_features // N`. For M3 we assume even division; uneven tiling with remainder handling is deferred to M4+.
- Tile PEs are placed sequentially. Collect PE follows.

**Expanded node naming:** A `LINEAR` node `"linear1"` expands to placed nodes `"linear1_tile_0"`, `"linear1_tile_1"`, ..., `"linear1_tile_N-1"`, and `"linear1_collect"`. The placer also generates internal edges from each tile node to the collect node. These are regular `PlacedEdge` entries in the `SpatialIR`.

**Tile metadata:** `PlacedNode` gains an optional `attrs: dict[str, Any] | None = None` field. Tile nodes carry `attrs={"tile_index": i, "rows_per_pe": r, "in_features": k, "origin_id": "linear1"}`. The collect node carries `attrs={"num_fragments": N, "origin_id": "linear1"}`. The `origin_id` links expanded nodes back to the original graph node for input slot mapping.

**For M3, we assume the graph contains a single `LINEAR` node.** Multi-operator placement (mixing LINEAR with FORWARD/COLLECT, or multiple LINEARs) is deferred to M4+.

For `FORWARD` and `COLLECT` nodes, behavior is unchanged from M2.

### 2.4 Routing pass

For each tile PE of a `LINEAR` node, the router creates:

- A `LinearTask` entry with `weight_slot=1`, `bias_slot=2`, shape info, and a route to the collect node. Each tile PE routes its output fragment to a **distinct payload slot** on the collect PE (tile index = payload slot), avoiding SRAM overwrites.
- The collect PE gets N `ConcatCollectTask` entries — one per fragment slot — each with `trigger_slot=i` (for `i` in `0..N`) and `num_fragments=N`.

**Input broadcast:** The input slot for a `LINEAR` node uses the `origin_id` as the name and maps to multiple coordinates (one per tile PE). The artifact's `input_slots` list contains multiple entries with the same `name` but different `coord` values. On the Rust side, `run_with_inputs` sends the same payload to all matching coords.

### 2.5 Weight data flow through passes

Weight data must reach the lowering pass to populate `initial_sram`. The flow:

1. `compile()` receives `weights` dict and validates shapes against node attrs.
2. `compile()` passes `weights` to the routing pass (or a post-route step) which tiles the weight matrix and attaches tile data to each `PESchedule`.
3. `PESchedule` gains an `initial_sram: dict[int, list[float]]` field (default empty).
4. The routing pass populates `initial_sram` on each tile PE's schedule: slot 1 = weight tile, slot 2 = bias tile.
5. The lowering pass mechanically copies `initial_sram` from `PESchedule` to `PEProgram` (1:1 translation, no tiling logic in lower).

**Weight tiling:** For tile index `i` with `rows_per_pe` rows, the weight chunk is `W[i*rows_per_pe : (i+1)*rows_per_pe, :]` flattened row-major. Bias chunk is `b[i*rows_per_pe : (i+1)*rows_per_pe]`.

---

## 3. Artifact schema changes

### 3.1 Per-kind task dataclasses

Replace the single `TaskProgram` with per-kind dataclasses. Each serializes as a flat dict with `kind` as the discriminator:

```python
@dataclass
class ForwardActivationTask:
    kind: str = field(default="forward_activation", init=False)
    trigger_slot: int
    input_slot: int
    route_dest: tuple[int, int]
    route_hops: list[str]

@dataclass
class CollectOutputTask:
    kind: str = field(default="collect_output", init=False)
    trigger_slot: int
    input_slot: int

@dataclass
class LinearTask:
    kind: str = field(default="linear", init=False)
    trigger_slot: int
    input_slot: int
    weight_slot: int
    bias_slot: int
    tile_rows: int          # local output dim (rows in this PE's weight tile)
    tile_cols: int          # input dim (= full in_features)
    route_dest: tuple[int, int]
    route_hops: list[str]
    fragment_slot: int      # payload slot on the collect PE for this tile's output

@dataclass
class ConcatCollectTask:
    kind: str = field(default="concat_collect", init=False)
    trigger_slot: int       # one per fragment slot; compiler generates N of these
    num_fragments: int      # total expected fragments
    rows_per_fragment: int  # size of each fragment (for offset calculation)
```

`PEProgram.tasks` becomes `list[ForwardActivationTask | CollectOutputTask | LinearTask | ConcatCollectTask]`.

### 3.2 Serialization

`_program_to_dict` converts each task dataclass to a dict with all fields, including `kind`. `_dict_to_program` matches on the `kind` field and constructs the corresponding dataclass from the remaining dict fields via keyword arguments (e.g., `LinearTask(**{k: v for k, v in d.items() if k != "kind"})`). This is a refactor of the existing code, not a new mechanism.

### 3.3 Backward compatibility

Existing artifacts with `forward_activation` and `collect_output` tasks continue to work. The `kind` field was already present in the M2 schema — we're just making the surrounding fields vary by kind instead of having a single struct with `Option` fields.

---

## 4. Rust runtime changes

### 4.1 New task kind

```rust
pub enum TaskKind {
    ForwardActivation {
        input_slot: SlotId,
        route_dest: Coord,
        hops: Vec<Direction>,
    },
    CollectOutput {
        input_slot: SlotId,
    },
    Linear {                          // new
        input_slot: SlotId,
        weight_slot: SlotId,
        bias_slot: SlotId,
        tile_rows: u32,               // local output dim
        tile_cols: u32,               // input dim (= full in_features)
        route_dest: Coord,
        hops: Vec<Direction>,
        fragment_slot: SlotId,        // payload slot on collect PE
    },
    ConcatCollect {                   // new
        num_fragments: u32,
        rows_per_fragment: u32,
    },
}
```

### 4.2 Execution logic

In `process_execute()`, two new arms:

**`Linear` arm:**

1. Reads input activation from `input_slot` — `Vec<f32>` of length `tile_cols`.
2. Reads weight tile from `weight_slot` — `Vec<f32>` of length `tile_rows * tile_cols`, row-major.
3. Reads bias tile from `bias_slot` — `Vec<f32>` of length `tile_rows`.
4. Computes `y[i] = sum(W[i * tile_cols + j] * x[j] for j in 0..tile_cols) + b[i]` for each `i in 0..tile_rows`.
5. Routes output fragment to `route_dest` via `hops`, delivering to `fragment_slot` on the collect PE.

Pure nested loops, no external math libraries. `tile_rows` / `tile_cols` are the local tile dimensions.

**`ConcatCollect` arm:**

1. Reads the incoming fragment from the trigger slot.
2. If the accumulator buffer doesn't exist yet, allocates `vec![0.0; num_fragments * rows_per_fragment]` in a designated SRAM slot.
3. Writes the fragment into the buffer at offset `trigger_slot * rows_per_fragment`.
4. Increments an arrival counter. When counter == `num_fragments`, moves the completed buffer to `self.outputs` keyed by the PE's coordinate (same convention as `CollectOutput`).

### 4.3 Artifact deserialization

Replace the current `TaskProgram` struct + manual `convert_task()` with a serde tagged enum:

```rust
#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
enum TaskProgram {
    #[serde(rename = "forward_activation")]
    ForwardActivation {
        trigger_slot: u32,
        input_slot: u32,
        route_dest: (u32, u32),
        route_hops: Vec<String>,
    },
    #[serde(rename = "collect_output")]
    CollectOutput {
        trigger_slot: u32,
        input_slot: u32,
    },
    #[serde(rename = "linear")]
    Linear {
        trigger_slot: u32,
        input_slot: u32,
        weight_slot: u32,
        bias_slot: u32,
        tile_rows: u32,
        tile_cols: u32,
        route_dest: (u32, u32),
        route_hops: Vec<String>,
        fragment_slot: u32,
    },
    #[serde(rename = "concat_collect")]
    ConcatCollect {
        trigger_slot: u32,
        num_fragments: u32,
        rows_per_fragment: u32,
    },
}
```

This eliminates the manual `convert_task()` dispatch function — serde handles deserialization directly from the `kind` discriminator. The `parse_direction()` helper is still needed to convert hop strings to `Direction` enums.

**Backward compatibility note:** Serde's internally tagged enum ignores unknown fields by default, so old-style artifacts that include `route_dest: null` and `route_hops: null` on `collect_output` tasks will still deserialize. Do NOT use `#[serde(deny_unknown_fields)]` on the enum variants.

### 4.4 Broadcast input slots

`LoadedProgram.input_slots` changes from `HashMap<String, InputSlotInfo>` to `HashMap<String, Vec<InputSlotInfo>>`. When `run_with_inputs` encounters a name, it sends the same payload to every entry in the vec.

The duplicate-name check in `load_program` is replaced: instead of rejecting duplicates, it groups entries by name. The `ProgramError::DuplicateInputSlot` variant is removed, as duplicate input slot names are now valid and represent broadcast targets.

---

## 5. Reference implementation and testing

### 5.1 Python reference

Add `torch` as a dev dependency (`pyproject.toml` dev group).

Create `python/meshflow/models/reference.py`:

```python
import torch

def reference_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)
```

### 5.2 End-to-end test

Test case dimensions: `in_features=4, out_features=6, num_tiles=3` (mesh 4x1).

```python
def test_linear_matches_torch():
    torch.manual_seed(42)
    in_f, out_f = 4, 6
    W = torch.randn(out_f, in_f)
    b = torch.randn(out_f)
    x = torch.randn(in_f)

    # Compile and run on simulator
    graph = GraphIR(
        nodes=[Node(id="linear1", op=OpType.LINEAR, attrs={"in_features": in_f, "out_features": out_f})],
        edges=[],
    )
    config = CompilerConfig(mesh_width=4)
    program = compile(graph, config, weights={"linear1": {"weight": W.numpy(), "bias": b.numpy()}})
    artifact_bytes = serialize(program)
    result = run_program(artifact_bytes, inputs={"linear1": x.tolist()})

    # Compare against torch reference
    expected = torch.nn.functional.linear(x, W, b)
    actual = torch.tensor(result.outputs[(3, 0)])
    assert torch.allclose(actual, expected, atol=1e-6)
```

### 5.3 Rust unit tests

- `Linear` task with hand-computed values (e.g., 2x2 matmul).
- `ConcatCollect` task with fragment ordering verification.
- Artifact deserialization with tagged enum (all four task kinds).
- `compiler_test_helpers` updated with `make_linear_artifact()`.

### 5.4 Tolerance

`atol=1e-6` — same operations on same-size f32 values, small matrices, no significant accumulation drift.

---

## 6. Scope boundaries

**In scope:**
- `OpType.LINEAR` with `attrs` for shape
- `weights` parameter on `compile()`
- Row-wise weight tiling across PEs
- Per-kind task dataclasses in artifact (Python) and serde tagged enum (Rust)
- `TaskKind::Linear` with matmul + bias in Rust
- `TaskKind::ConcatCollect` for ordered fragment concatenation
- Broadcast input slots (one name → multiple PEs)
- `torch` as dev dependency
- End-to-end correctness test against torch

**Out of scope (M4+):**
- Multiple layers / activation functions
- Column-wise or 2D tiling
- Inter-layer activation pipelining
- Uneven tiling / remainder handling
- Torch model import (`torch.nn.Module` → GraphIR)
- Operator-specific latency modeling (all tasks still cost `task_base_latency`)
