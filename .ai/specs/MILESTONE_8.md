# Milestone 8: Codebase Quality + Cost Model

## Goal

Improve internal code quality through test hardening, runtime/compiler refactoring, and a data-proportional cost model. No new operators or model capabilities — this milestone strengthens the foundation built in M1–M7.

## Sequencing

Three phases, strictly ordered:

1. **Test infrastructure** — MLP model helper + operator e2e tests. Establishes a safety net.
2. **Refactoring** — Extract helpers from runtime.rs and route.py. Pure structural changes verified by the test suite.
3. **Cost model** — Data-proportional task costs that make profiling output reflect actual computational work.

---

## Phase 1: Test Infrastructure

### MLP model helper

**File**: `python/meshflow/models/mlp.py`

```python
def mlp_block(layer_dims: list[int]) -> GraphIR:
```

Constructs an MLP graph from a list of layer dimensions. Example: `mlp_block([4, 8, 4])` creates:

```
FORWARD → Linear(4→8) → ReLU → Linear(8→4) → COLLECT
```

General pattern for N layers: FORWARD → (Linear → ReLU) × (N-1) → Linear → COLLECT. No ReLU after the final layer.

Node IDs follow a consistent scheme: `"input"`, `"linear0"`, `"relu0"`, `"linear1"`, `"output"`. Edges connect sequentially with `src_slot=0, dst_slot=0`.

```python
def mlp_weights(layer_dims: list[int], seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
```

Generates random weights keyed by node ID, matching the graph from `mlp_block()`. Uses Xavier-like initialization (`scale = 1/sqrt(in_features)`). Zero biases. Same pattern as `transformer_weights()`.

**No new reference implementation needed** — `reference_mlp()` already exists in `reference.py`.

### Operator e2e tests

**File**: `tests/python/runtime/test_operators.py`

Each test class compiles a graph through the full pipeline (`compile()` → `serialize()` → `run_program()`) and validates numerical correctness against the torch reference.

A shared `_run_graph()` helper encapsulates the compile-serialize-run-extract pattern used by all tests.

#### TestRmsNorm

Minimal graph: `FORWARD → RMSNORM → COLLECT`. This compiles and runs correctly as a standalone graph.

- `test_single_position`: d_model=8, single input vector. Compare against `reference_rmsnorm()`. Tolerance: `atol=1e-4`.
- `test_multi_position`: seq_len=3, d_model=8. Three positions normalized independently. Compare per-position against `reference_rmsnorm()`.
- `test_non_divisible_features`: d_model=7, mesh_height=5. Exercises uneven tile distribution in RmsNorm slicing.

#### TestSoftmax

**Cannot be tested standalone.** The route pass only generates Softmax tasks co-located within attention chains — a standalone `FORWARD → SOFTMAX → COLLECT` graph compiles but the Softmax output is stranded in PE-local SRAM with no forwarding task. Tests exercise Softmax through a transformer block attention chain.

- `test_softmax_in_attention_chain`: Small transformer block (seq_len=2, d_model=4). Full chain vs `reference_transformer_block()`. Validates that the attention sublayer (which includes softmax) produces correct results.
- `test_softmax_numerical_stability`: Transformer block with large-scale Q/K weights (×100) to push QKT scores into overflow range. Verify no NaN/Inf in output.

#### TestAdd

Graph: Two FORWARD nodes → ADD → COLLECT. The two FORWARD nodes feed into `dst_slot=0` and `dst_slot=1` on the ADD node. This compiles and runs correctly as a standalone graph.

- `test_basic`: Two vectors of length 4. Verify element-wise sum with `torch.allclose`.
- `test_multi_position`: seq_len=2, d_model=4. Two 8-element vectors added element-wise.

#### TestMatMul

**Cannot be tested standalone.** Same issue as Softmax — the route pass only generates MatMul tasks co-located within attention chains. A standalone MATMUL's output is stranded in PE-local SRAM. Tests exercise MatMul through transformer block attention chains.

- `test_attention_matmul_small`: seq_len=2, d_model=4, d_ff=8. Full block vs torch reference. Exercises both QKT (K @ q_row, transpose=false) and AV (V^T @ weights, transpose=true) MatMul operations.
- `test_attention_matmul_larger`: seq_len=4, d_model=8, d_ff=16. Larger dimensions for coverage.

#### TestMlpHelper

Uses the new `mlp_block()` and `mlp_weights()` helpers:

- `test_two_layer`: `mlp_block([4, 8, 4])` with random weights. Compare against `reference_mlp()`.
- `test_three_layer`: `mlp_block([4, 8, 16, 4])` with random weights. Compare against `reference_mlp()`.

---

## Phase 2: Refactoring

### Rust runtime.rs

`process_execute` is ~595 lines with 9 `Message { ... }` construction sites and 6 near-identical broadcast loops. Three helpers extracted:

#### `emit_message()`

Creates a Message, increments `next_message_id`, enqueues delivery.

```rust
fn emit_message(
    &mut self,
    timestamp: u64,
    source: Coord,
    dest: Coord,
    hops: Vec<Direction>,
    payload: Vec<f32>,
    payload_slot: SlotId,
)
```

Replaces the repeated pattern:
```rust
let message = Message {
    id: self.next_message_id,
    source: coord,
    dest: *dest,
    hops: hops.clone(),
    current_hop: 0,
    payload: ...,
    payload_slot: ...,
    timestamp,
};
self.next_message_id += 1;
self.enqueue_deliver(timestamp, coord, coord, message);
```

Used by: ForwardActivation, Linear, RmsNormPartialSum, and as the building block for broadcast/scatter helpers.

#### `broadcast_to_dests()`

Loops over destination list with serialized send times, calls `emit_message` for each. Broadcasts the full payload to every destination. Handles `messages_sent` counter internally.

```rust
fn broadcast_to_dests(
    &mut self,
    base_time: u64,
    coord: Coord,
    dests: &[(Coord, Vec<Direction>)],
    payload_slots: &[SlotId],
    payload: Vec<f32>,
)
```

Replaces the identical broadcast loop in: Add, MatMul, RmsNormNormalize, RmsNormReduce, and the broadcast branch of ConcatCollectForward.

**ConcatCollectForward note**: The existing code at line ~484 increments `messages_sent` before the scatter/broadcast block. This line must be **removed** — `broadcast_to_dests`/`scatter_to_dests` handle the counter internally. The activation application logic (lines ~477-480) stays unchanged; only the `messages_sent` increment and the scatter/broadcast block get replaced.

#### `scatter_to_dests()`

Like broadcast but slices the result vector — row i goes to destination i.

```rust
fn scatter_to_dests(
    &mut self,
    base_time: u64,
    coord: Coord,
    dests: &[(Coord, Vec<Direction>)],
    payload_slots: &[SlotId],
    result: &[f32],
)
```

Replaces the scatter branch in ConcatCollectForward.

#### What stays in process_execute

The match arms themselves stay inline — they contain the operator-specific compute logic (matrix multiply, softmax, normalization, etc.) which is not duplicated. Only the message creation and routing boilerplate is extracted.

### Python route.py

Two small helper extractions:

#### `_outgoing_edges(spatial, node_id)`

Replaces the 3 identical `[e for e in spatial.edges if e.src_node == node.id]` list comprehensions at lines 66, 86, 129.

```python
def _outgoing_edges(spatial: SpatialIR, node_id: str) -> list[PlacedEdge]:
    return [e for e in spatial.edges if e.src_node == node_id]
```

`PlacedEdge` must be added to the existing import from `spatial_ir` (line 24-33).

#### `_load_linear_weights()`

Extracts the weight-slicing logic (weight matrix row selection + bias slice) into a dedicated function for readability. The weight-loading code runs once per LINEAR_TILE node inside the main loop — this extraction separates weight-loading concerns from task setup, not deduplication of multiple source sites.

```python
def _load_linear_weights(
    pe_sram: dict,
    coord: tuple[int, int],
    weights: dict[str, dict[str, np.ndarray]] | None,
    origin_id: str,
    fragment_offset: int,
    tile_rows: int,
    weight_slot: int = 1,
    bias_slot: int = 2,
) -> None:
```

### Validation

All existing tests (91 Rust, 252+ Python) must pass unchanged after Phase 2. The Phase 1 operator tests provide additional coverage during refactoring. No behavioral changes — pure structural improvement.

---

## Phase 3: Data-Proportional Cost Model

### SimConfig changes

```rust
pub struct SimConfig {
    pub width: u32,
    pub height: u32,
    pub hop_latency: u64,
    pub task_base_latency: u64,
    pub cost_per_element: u64,   // NEW — cost per multiply-accumulate (default: 1)
    pub max_events: u64,
}
```

`task_base_latency` remains as fixed overhead per task (message setup, SRAM reads). `cost_per_element` is the variable component scaling with work performed.

Total task cost = `task_base_latency + elements * cost_per_element`.

### Helper

```rust
fn task_cost(&self, elements: u64) -> u64 {
    self.config.task_base_latency + elements * self.config.cost_per_element
}
```

### Per-operator element counts

| Operator | Elements | Source of dimensions |
|----------|----------|---------------------|
| Linear | `tile_rows * tile_cols * num_positions` | TaskKind fields + `x.len() / tile_cols` |
| MatMul (no transpose) | `rows * cols` | TaskKind fields |
| MatMul (transpose) | `rows * cols` | TaskKind fields |
| RmsNormPartialSum | `slice_size * num_positions` | TaskKind fields + inferred from data |
| RmsNormNormalize | `slice_size * num_positions` | TaskKind fields + inferred from data |
| RmsNormReduce | `num_tiles * num_positions` | TaskKind fields + inferred from data |
| Softmax | `3 * row_length` | `data.len()` (3 passes: max, exp, normalize) |
| Add | `data.len()` | From SRAM slot |
| ForwardActivation | 0 | Pure data movement |
| CollectOutput | 0 | Pure data movement |
| ConcatCollect | 0 | Accumulation bookkeeping only |
| ConcatCollectForward | 0 | Accumulation bookkeeping only |

### Cost application and triggering semantics

There are two paths that trigger task execution:

1. **Message delivery** (`process_deliver`): When a message arrives, the triggered task is scheduled at `timestamp + task_base_latency`. This represents the startup overhead from message arrival to task execution.
2. **Co-located task write** (`task_write_slot`): When a task writes to a local SRAM slot (e.g., Softmax writing scores that trigger AV MatMul), the triggered task is scheduled at `timestamp + task_base_latency`.

**Both paths keep `task_base_latency` as the startup overhead. Neither path changes.**

The element cost is applied by the *producing* task to delay its output:

- **Operators that emit messages** (Linear, RmsNormPartialSum): Use `timestamp + task_cost(elements)` as the message send time. Full cost = `task_base_latency + elements * cost_per_element`.
- **Operators that broadcast/scatter** (Add, MatMul, RmsNormNormalize, RmsNormReduce): Use `timestamp + task_cost(elements)` as the base time for `broadcast_to_dests`.
- **Operators that write locally then broadcast** (Add): Pass `timestamp + elements * cost_per_element` to `task_write_slot` (which adds `task_base_latency` internally for triggered tasks). Use `timestamp + task_cost(elements)` as the base time for `broadcast_to_dests`.
- **Operators that write locally only** (Softmax): Pass `timestamp + elements * cost_per_element` to `task_write_slot`. The triggered co-located task (AV MatMul) starts at `timestamp + element_cost + task_base_latency`.

### Threading `cost_per_element` through the stack

Four files need the new field:

1. **`runtime.rs`**: `SimConfig` struct + `Default` impl + `task_cost()` helper.
2. **`bridge.rs`**: `MeshConfig` PyO3 class (field + constructor with `cost_per_element=1` default) + `From<&MeshConfig> for SimConfig` impl.
3. **`program.rs`**: `MeshProgramConfig` serde struct (with `#[serde(default = "default_cost_per_element")]` for backward compat) + `SimConfig` construction in `load_program()`.
4. **`artifact.py`**: Python `MeshProgramConfig` dataclass (with `cost_per_element: int = 1`).

The `CompilerConfig` on the Python side does not need this field — it's a runtime parameter, not a compilation parameter.

### Backward compatibility

Default `cost_per_element = 1` preserves existing behavior for simple cases. Existing Rust tests use `..Default::default()` for SimConfig and will pick up the new default. JSON test literals in program.rs omit `cost_per_element` but the `#[serde(default)]` annotation handles this. Tests that assert on exact timestamp values may need updating if the data-proportional cost changes their operator timings.

### New tests

- `test_linear_cost_scales_with_dimensions`: Compile two LINEAR graphs (4→4 vs 4→16) with different dimensions. Verify the larger one has proportionally higher `final_timestamp`.
- `test_matmul_cost_scales_with_matrix_size`: Compile two transformer blocks (small vs large dimensions). Verify the larger one has higher `final_timestamp`.
- `test_forward_activation_zero_element_cost`: Verify a FORWARD→COLLECT chain has low `final_timestamp` (no element cost, only hop + task overhead).
- `test_cost_per_element_configurable`: Run the same graph via `run_simulation` with `cost_per_element=1` vs `cost_per_element=10`. Verify the plumbing works (for zero-element tasks like CollectOutput, both should produce the same timestamp).

---

## Out of Scope

- **Rust error handling changes** — Panics are appropriate for invariant violations in a simulator.
- **Golden/snapshot artifact tests** — Existing round-trip + numerical e2e tests catch meaningful regressions. Golden tests add maintenance burden without much incremental value.
- **Congestion-aware cost model** — Queue-depth penalties would change simulator semantics and add event loop complexity. Data-proportional costs are sufficient.
- **New operators or model capabilities** — This milestone is purely about internal quality.
- **Standalone Softmax/MatMul routing** — These operators only work co-located within attention chains. Adding standalone forwarding is possible but out of scope.

---

## Exit Criteria

- `mlp_block()` and `mlp_weights()` model helper works and is tested.
- E2e tests exist for RmsNorm, Softmax (via attention chain), Add, MatMul (via attention chain), and MLP helper.
- `process_execute` uses `emit_message()`, `broadcast_to_dests()`, and `scatter_to_dests()` helpers.
- `route.py` uses `_outgoing_edges()` and `_load_linear_weights()` helpers.
- All existing tests pass unchanged after refactoring (no behavioral changes).
- Task costs scale with operator work: Linear and MatMul costs reflect matrix dimensions.
- `cost_per_element` is configurable via the PyO3 bridge.
- All lints clean, all tests pass.
