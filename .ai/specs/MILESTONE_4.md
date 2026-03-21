# Milestone 4: Multi-Layer MLP Pipeline

## Objective

Execute a multi-layer MLP (e.g., `Linear → ReLU → Linear → ReLU → Linear`) distributed across a 2D mesh of PEs. Each LINEAR layer occupies a column; tiles stack vertically within a column, activations flow horizontally between columns. ReLU is fused onto the collect PE of the preceding layer. Correctness is verified against a PyTorch reference MLP.

This is the first milestone with inter-layer data flow — activations produced by one layer become inputs to the next.

### Coordinate convention

Coordinates are `(x, y)` where x increases eastward (columns) and y increases northward (rows). North = +y, South = -y, East = +x, West = -x. This matches the existing `coords.rs` convention. In layout diagrams, row 0 is at the bottom (y=0) and higher rows are at higher y values.

## Exit criteria

1. A 2-layer MLP (`Linear → ReLU → Linear`) compiles, serializes, loads, and runs end-to-end.
2. A 3-layer MLP also works (proves N-layer generality).
3. Simulator output matches `reference_mlp` within `atol=1e-6`.
4. Uneven tiling (out_features not divisible by num_tiles) produces correct results.
5. Profiling shows send serialization costs in timing.
6. All existing M3 tests pass (with config migration).
7. All linters clean (`cargo fmt`, `clippy`, `ruff`, `mypy`).

---

## 1. Data flow overview

For a 2-layer MLP `Linear(4,8) → ReLU → Linear(8,3)` on a mesh with `mesh_height=4`:

```
           col 0 (layer 1)       col 1 (layer 2)       (x=0)          (x=1)
y=3        L1_collect+ReLU       L2_collect (terminal)
y=2        L1_tile2              L2_tile2
y=1        L1_tile1              L2_tile1
y=0        L1_tile0              L2_tile0
```

**Flow:**

1. Input `x (4,)` is broadcast to L1 tile PEs (col 0, y=0–2).
2. Each L1 tile computes its fragment of `W1 @ x + b1`, routes north to L1 collect at `(0, 3)`.
3. L1 collect assembles fragments, applies ReLU, then broadcasts to each L2 tile PE: east + south hops (send serialization: 3 sends = 3 time units).
4. Each L2 tile computes its fragment of `W2 @ relu_out + b2`, routes north to L2 collect at `(1, 3)`.
5. L2 collect assembles fragments → final output.

### 1.1 2D column-per-layer layout

Each LINEAR layer gets one column. Within a column:

- Tile PEs occupy rows `0..num_tiles-1`.
- Collect PE occupies row `num_tiles`.
- `mesh_height` is a fixed config parameter. `num_tiles = min(mesh_height - 1, out_features)`.
- `mesh_width` is auto-determined: one column per LINEAR layer.
- Columns with fewer tiles than `mesh_height - 1` leave upper rows as unused PEs.

### 1.2 Uneven tiling

When `out_features` is not evenly divisible by `num_tiles`:

- `base = out_features // num_tiles`
- `remainder = out_features % num_tiles`
- Tiles `0..remainder-1` get `base + 1` rows; tiles `remainder..num_tiles-1` get `base` rows.

Each tile knows its own `tile_rows` and `fragment_offset` (byte offset into the output buffer). The offset for tile `i` is:

```
offset(i) = i * base + min(i, remainder)
```

Fragment payloads are self-sized (`len(payload)` determines how many values to copy at the offset).

### 1.3 ReLU fusion

`RELU` is a first-class `OpType` in GraphIR but does not produce its own PE. During placement, when a LINEAR node's sole successor is a RELU node, the RELU is fused onto the LINEAR's collect PE. The collect PE applies the activation after assembling all fragments and before broadcasting to the next layer.

### 1.4 Fragment collection (updated from M3)

M3's `rows_per_fragment` field is replaced with:

- `total_rows: int` — total output buffer size (= `out_features`), used to pre-allocate the accumulator.
- `fragment_offset: int` — where in the buffer this fragment's data starts.

The accumulator model is otherwise unchanged: O(1) SRAM slots (accum at `u32::MAX`, counter at `u32::MAX-1`), fragment slots freed after copy.

### 1.5 Send serialization

When a task emits multiple messages, each send costs 1 time unit. A collect PE broadcasting to 3 tile PEs:

- Send 0 at time T
- Send 1 at time T+1
- Send 2 at time T+2

This models the physical constraint that a PE can only put one message on the network per cycle. The simulator's `process_execute` returns events with staggered timestamps.

### 1.6 Inter-layer routing

The collect PE of an intermediate layer routes to each tile PE of the next layer. Routes are XY dimension-ordered: east to the next column, then south to the target row. For example, L1 collect at `(0, 3)` routing to L2 tile at `(1, 0)`: 1 hop east, 3 hops south.

---

## 2. GraphIR changes

### 2.1 New OpType

```python
class OpType(Enum):
    FORWARD = "forward"
    COLLECT = "collect"
    LINEAR = "linear"
    RELU = "relu"        # new
```

### 2.2 RELU node validation

- No `attrs` required.
- Must have exactly one incoming edge.
- Must have exactly one outgoing edge (to a LINEAR node) or zero outgoing edges (if applied to the final layer — though this is unusual).
- Must follow a LINEAR node (for M4; could relax later).

### 2.3 Shape chaining validation

The compiler validates that connected LINEAR layers have compatible dimensions: for each `LINEAR(out=K) → ... → LINEAR(in=K)` connection, `layer1.out_features == layer2.in_features`.

### 2.4 Example graph

```python
graph = GraphIR(
    nodes=[
        Node(id="linear1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 8}),
        Node(id="relu1", op=OpType.RELU),
        Node(id="linear2", op=OpType.LINEAR, attrs={"in_features": 8, "out_features": 3}),
    ],
    edges=[
        Edge(src_node="linear1", src_slot=0, dst_node="relu1", dst_slot=0),
        Edge(src_node="relu1", src_slot=0, dst_node="linear2", dst_slot=0),
    ],
)
```

---

## 3. CompilerConfig migration

### 3.1 New config shape

```python
@dataclass
class CompilerConfig:
    placement: PlacementStrategy = PlacementStrategy.SEQUENTIAL
    routing: RoutingStrategy = RoutingStrategy.DIMENSION_ORDERED_XY
    mesh_height: int = 4      # fixed, user-chosen; controls parallelism
    # mesh_width auto-determined from graph (one column per LINEAR layer)
```

The `mesh_width` field is removed. For single-layer M3 graphs, `mesh_height` replaces `mesh_width`: a single LINEAR column with `mesh_height=4` gives 3 tiles + 1 collect. The mesh becomes `1 x mesh_height` (one column) instead of M3's `mesh_width x 1` (one row).

**Coordinate migration:** This is a 90-degree rotation from M3's layout. A single LINEAR with 3 tiles that was at `(0,0), (1,0), (2,0), (3,0)` (M3) becomes `(0,0), (0,1), (0,2), (0,3)` (M4). All M3 test coordinate assertions (e.g., `result.outputs[(3, 0)]`) must be updated to the new column layout (e.g., `result.outputs[(0, 3)]`).

---

## 4. Placement pass changes

### 4.1 Algorithm

1. Walk graph in topological order.
2. Detect LINEAR→RELU pairs: if a LINEAR node's sole successor is a RELU node, mark the RELU as fused. Record `activation: "relu"` on the collect PE's attrs.
3. Assign columns left-to-right, one per LINEAR node. RELU nodes that are fused produce no placed nodes.
4. Per column with `out_features=F`:
   - `num_tiles = min(mesh_height - 1, out_features)`
   - `base = F // num_tiles`, `remainder = F % num_tiles`
   - Tile `i` gets `base + 1` rows if `i < remainder`, else `base` rows.
   - Tile PEs at `(col, 0)` through `(col, num_tiles - 1)`.
   - Collect PE at `(col, num_tiles)`.
5. Mesh dimensions: `width = num_columns`, `height = max(num_tiles + 1)` across all columns, padded up to `mesh_height` if smaller.
6. Internal edges: each tile → its column's collect PE (for fragment routing).

### 4.2 Placed node metadata

**Tile PE attrs:**

```python
{
    "tile_index": i,
    "tile_rows": rows_for_this_tile,
    "fragment_offset": offset_for_this_tile,
    "in_features": K,
    "origin_id": "linear1",
}
```

**Collect PE attrs (intermediate layer with fused ReLU):**

```python
{
    "num_fragments": N,
    "total_rows": out_features,
    "origin_id": "linear1",
    "activation": "relu",            # present only if RELU fused
    "route_to": "linear2",           # ID of next LINEAR node
}
```

**Collect PE attrs (terminal layer):**

```python
{
    "num_fragments": N,
    "total_rows": out_features,
    "origin_id": "linear1",
}
```

### 4.3 FORWARD/COLLECT nodes

FORWARD/COLLECT-only graphs continue to work via the existing M3 placement logic (the placement pass falls through to sequential 1D layout when no LINEAR nodes are present). Mixed graphs containing both FORWARD/COLLECT and LINEAR nodes in the same graph are not supported in M4.

---

## 5. Routing pass changes

### 5.1 Tile PE routing (same as M3, updated for uneven tiling)

For each tile PE of a LINEAR node:

- `LinearEntry` with `weight_slot=1`, `bias_slot=2`.
- `tile_rows` and `tile_cols` reflect this specific tile's dimensions.
- `fragment_offset` is the offset into the output buffer for this tile.
- Route: north hops from `(col, tile_row)` to `(col, collect_row)` (increasing y).
- `fragment_slot = tile_index` (distinct payload slots on collect PE).

### 5.2 Terminal collect PE (final layer)

N `ConcatCollectEntry` tasks, one per fragment slot:

```python
ConcatCollectEntry(
    trigger_slot=i,
    num_fragments=N,
    total_rows=out_features,
    fragment_offset=offset(i),
)
```

Same as M3 but with `fragment_offset` + `total_rows` replacing `rows_per_fragment`.

### 5.3 Intermediate collect PE (with inter-layer routing)

N `ConcatCollectForwardEntry` tasks, one per fragment slot:

```python
ConcatCollectForwardEntry(
    trigger_slot=i,
    num_fragments=N,
    total_rows=out_features,
    fragment_offset=offset(i),
    activation="relu",   # or None if no fused activation
    route_dests=[(coord, hops) for each tile PE in next layer],
)
```

When the last fragment arrives and the buffer is complete:
1. Apply activation (if specified).
2. Send the activated output to each destination in `route_dests` (with send serialization).

Routes from collect PE to next layer's tiles are XY dimension-ordered: east to next column, then north/south to target row.

### 5.4 Input slots

Only the **first layer's** tile PEs are external input slots. Intermediate layers receive input from the preceding collect PE's broadcast. Input slot name = `origin_id` of the first LINEAR node.

### 5.5 Weight slicing (updated for uneven tiling)

For tile `i` with `tile_rows` rows starting at row offset `fragment_offset`:

- `weight_tile = W[fragment_offset : fragment_offset + tile_rows, :]` flattened row-major.
- `bias_tile = b[fragment_offset : fragment_offset + tile_rows]`.

---

## 6. Artifact schema changes

### 6.1 Updated ConcatCollect task

Replace `rows_per_fragment` with `fragment_offset` + `total_rows`:

```python
@dataclass
class ConcatCollectTask:
    kind: str = field(default="concat_collect", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    total_rows: int = 0          # output buffer size
    fragment_offset: int = 0     # where this fragment writes
```

### 6.2 New ConcatCollectForward task

```python
@dataclass
class ConcatCollectForwardTask:
    kind: str = field(default="concat_collect_forward", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    total_rows: int = 0
    fragment_offset: int = 0
    activation: str | None = None               # "relu" or None
    route_dests: list[tuple[tuple[int, int], list[str]]] = field(default_factory=list)
        # list of (coord, hops) pairs
```

### 6.2.1 Serialization helpers

The existing `_task_to_dict()` and `_dict_to_task()` in `artifact.py` must be updated to handle `ConcatCollectForwardTask` (new kind) and the changed fields on `ConcatCollectTask` (`total_rows` + `fragment_offset` replacing `rows_per_fragment`). Same pattern as existing task kinds: match on `kind` string, construct from remaining dict fields.

### 6.3 Updated TaskProgram union

```python
TaskProgram = (
    ForwardActivationTask
    | CollectOutputTask
    | LinearTask
    | ConcatCollectTask
    | ConcatCollectForwardTask    # new
)
```

### 6.4 LinearTask update

`LinearTask` gains `fragment_offset`:

```python
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
    fragment_offset: int = 0     # new: offset in output buffer
```

`fragment_offset` on `LinearTask` is informational — the actual offset logic lives in `ConcatCollect`/`ConcatCollectForward`. But including it makes the artifact self-documenting. It must also be present on the Rust `TaskProgram::Linear` serde variant (Section 7.5) so that deserialization does not reject the field.

---

## 7. Rust runtime changes

### 7.1 Activation enum

```rust
#[derive(Debug, Clone, Deserialize)]
pub enum Activation {
    #[serde(rename = "relu")]
    ReLU,
}

fn apply_activation(activation: &Activation, data: &mut [f32]) {
    match activation {
        Activation::ReLU => {
            for v in data.iter_mut() {
                *v = v.max(0.0);
            }
        }
    }
}
```

### 7.2 New task kind: ConcatCollectForward

```rust
ConcatCollectForward {
    num_fragments: u32,
    total_rows: u32,
    fragment_offset: u32,
    activation: Option<Activation>,
    route_dests: Vec<(Coord, Vec<Direction>)>,
}
```

The existing `Linear` variant also gains `fragment_offset: u32` (informational, included for artifact completeness).

### 7.2.1 Code sharing between ConcatCollect and ConcatCollectForward

The accumulator logic (buffer allocation, offset-based writes, counter tracking, fragment slot cleanup) is identical between `ConcatCollect` and `ConcatCollectForward`. Extract this into a shared helper (e.g., `process_concat_fragment`) that both task arms call. `ConcatCollectForward` adds activation + broadcast after the helper signals completion. Avoid duplicating the accumulator logic across the two arms.

**Execution logic:**

1. Same accumulator model as `ConcatCollect`: read fragment from trigger slot, write into buffer at `fragment_offset`, increment counter, free fragment slot.
2. When counter == `num_fragments`:
   - Apply activation (if `Some`).
   - For each `(dest_coord, hops)` in `route_dests`, send a message with the full activated buffer as payload. Each send scheduled at `current_time + i` (send serialization).
   - Free accumulator and counter slots.

### 7.3 Updated ConcatCollect

Replace `rows_per_fragment` with `total_rows` + `fragment_offset`:

```rust
ConcatCollect {
    num_fragments: u32,
    total_rows: u32,
    fragment_offset: u32,
}
```

Accumulator buffer pre-allocated to `total_rows` (instead of `num_fragments * rows_per_fragment`). Fragment written at `fragment_offset` for `payload.len()` values.

### 7.4 Send serialization

In `process_execute`, when a task produces multiple outbound messages, each message is scheduled with a staggered timestamp:

```rust
for (i, (dest, hops)) in route_dests.iter().enumerate() {
    let send_time = current_time + task_base_latency + i as u64;
    // enqueue DeliverMessage event at send_time
}
```

This applies to `ConcatCollectForward` (multi-destination broadcast) and any future task that sends multiple messages. Single-send tasks (Linear, ForwardActivation) are unaffected.

### 7.5 Serde tagged enum additions

The `Activation` enum (Section 7.1) derives `Deserialize`, so serde can deserialize `"relu"` directly to `Activation::ReLU`. The `Option<Activation>` field handles `null` / absent values as `None`.

```rust
#[serde(rename = "concat_collect_forward")]
ConcatCollectForward {
    trigger_slot: u32,
    num_fragments: u32,
    total_rows: u32,
    fragment_offset: u32,
    activation: Option<Activation>,
    route_dests: Vec<((u32, u32), Vec<String>)>,
},
```

The `route_dests` hop strings are converted to `Direction` enums using the existing `parse_direction` helper during `convert_task`.

---

## 8. ScheduleIR changes

### 8.1 New entry type

```python
@dataclass
class ConcatCollectForwardEntry:
    kind: str = field(default="concat_collect_forward", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    total_rows: int = 0
    fragment_offset: int = 0
    activation: str | None = None
    route_dests: list[tuple[tuple[int, int], list[Direction]]] = field(default_factory=list)
```

### 8.2 Updated ConcatCollectEntry

```python
@dataclass
class ConcatCollectEntry:
    kind: str = field(default="concat_collect", init=False)
    trigger_slot: int = 0
    num_fragments: int = 0
    total_rows: int = 0          # replaces rows_per_fragment
    fragment_offset: int = 0     # replaces rows_per_fragment
```

### 8.3 Updated TaskEntry union

```python
TaskEntry = (
    ForwardActivationEntry
    | CollectOutputEntry
    | LinearEntry
    | ConcatCollectEntry
    | ConcatCollectForwardEntry    # new
)
```

---

## 9. Reference implementation and testing

### 9.1 Python reference

Extend `python/meshflow/models/reference.py`:

```python
def reference_mlp(
    x: torch.Tensor,
    layers: list[tuple[torch.Tensor, torch.Tensor]],
    activation: str = "relu",
) -> torch.Tensor:
    """Reference MLP: chain of linear layers with activation between them."""
    for i, (W, b) in enumerate(layers):
        x = torch.nn.functional.linear(x, W, b)
        if i < len(layers) - 1:  # no activation on final layer
            x = torch.relu(x)
    return x
```

### 9.2 End-to-end tests

1. **2-layer MLP correctness:** `Linear(4,8) → ReLU → Linear(8,6)`, random weights, compare against `reference_mlp` with `atol=1e-6`.
2. **3-layer MLP correctness:** `Linear(4,8) → ReLU → Linear(8,6) → ReLU → Linear(6,3)`, verifies N-layer generality.
3. **Uneven tiling:** layer where `out_features % num_tiles != 0` (e.g., `out_features=7, mesh_height=4` → 3 tiles with rows 3,2,2), verify correct output.
4. **No activation on final layer:** last layer's collect is terminal `ConcatCollect`, not `ConcatCollectForward`.
5. **Single-tile degenerate:** `mesh_height=2` (1 tile + 1 collect per column), MLP still works.
6. **Profiling counters:** verify send serialization in timing — broadcast to 3 tiles adds 3 time units. Verify `total_messages` includes inter-layer traffic.

### 9.2.1 Validation error tests

- RELU node not preceded by a LINEAR node → compiler error.
- Shape mismatch: `Linear(out=8) → ReLU → Linear(in=6)` → compiler error.

### 9.3 Rust unit tests

- `ConcatCollectForward` with ReLU: hand-verified values (negative values zeroed).
- Send serialization: verify staggered timestamps in event queue.
- Uneven fragment offsets: verify accumulator assembly with different-sized fragments.
- Artifact deserialization with `concat_collect_forward` variant.
- Updated `compiler_test_helpers` with `make_mlp_artifact()`.

### 9.4 M3 regression

Existing single-layer LINEAR tests are updated for:
- `CompilerConfig(mesh_height=...)` replacing `CompilerConfig(mesh_width=...)`.
- `fragment_offset` + `total_rows` replacing `rows_per_fragment` on `ConcatCollect`.
- Coordinate assertions rotate from horizontal `(x, 0)` to vertical `(0, y)` layout (e.g., `result.outputs[(3, 0)]` becomes `result.outputs[(0, 3)]`).
- Rust `compiler_test_helpers` updated for new layout and ConcatCollect fields.

All existing M3 test scenarios must continue to pass with updated assertions.

### 9.5 Tolerance

`atol=1e-6` — same as M3. Small matrices, f32, no significant accumulation drift.

---

## 10. Scope boundaries

**In scope:**

- `OpType.RELU` in GraphIR
- ReLU fusion onto collect PE during placement
- 2D column-per-layer mesh layout with fixed `mesh_height`
- Uneven tiling (base/remainder distribution)
- `ConcatCollectForward` task kind (collect + activate + broadcast)
- `fragment_offset` + `total_rows` replacing `rows_per_fragment` on all ConcatCollect variants
- Send serialization (each send costs 1 time unit)
- Inter-layer routing (collect PE → next layer's tile PEs via XY routing)
- `reference_mlp` for torch comparison
- CompilerConfig migration: `mesh_width` → `mesh_height`
- Shape chaining validation in compiler

**Out of scope (deferred):**

- Activation functions beyond ReLU (sigmoid, GELU, softmax)
- Tree-based broadcast / multicast
- Column-wise or 2D weight tiling
- Torch model import (`nn.Sequential` → GraphIR)
- Operator-specific latency modeling (all tasks still cost `task_base_latency`)
- SRAM capacity enforcement
- Pipelining across batches (batch size remains 1)
- API layer (Milestone 5)
- Observability improvements (Milestone 6)
