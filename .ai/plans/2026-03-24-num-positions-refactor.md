# Multi-Position Support (`num_positions`) Refactor

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LINEAR, ConcatCollect, RMSNorm, and MATMUL batch-aware so they can process multiple sequence positions with the same weights, enabling seq_len > 1 transformer blocks.

**Architecture:** Each operator infers `num_positions` from its input size vs expected single-position size. Outputs are position-major (positions outer, features inner). No new task kinds — existing tasks gain multi-position capability while remaining backward compatible (num_positions=1 for existing MLP graphs).

**Tech Stack:** Rust (mesh_runtime), Python (compiler passes, schedule_ir, artifact)

---

## Chunk 1: Batched LINEAR + ConcatCollect

### Task 1: Batched LINEAR execution

**Files:**
- Modify: `crates/mesh_runtime/src/runtime.rs` (process_execute Linear arm)
- Test: `crates/mesh_runtime/src/runtime.rs` (tests module)

- [ ] **Step 1: Write failing test for batched LINEAR**

```rust
#[test]
fn linear_batched_two_positions() {
    // W = [[1, 0], [0, 1]], b = [0, 0] (2x2 identity)
    // Input: 2 positions of 2 features each = [1, 2, 3, 4]
    // Expected: [1, 2, 3, 4] (identity)
    // Fragment to collect PE, which should get [1, 2, 3, 4]
}
```

- [ ] **Step 2: Run test, verify it fails**

- [ ] **Step 3: Implement batched LINEAR**

In the `Linear` arm of `process_execute`:

```rust
let cols = tile_cols as usize;
let rows = tile_rows as usize;
let num_positions = x.len() / cols;

let mut y = Vec::with_capacity(rows * num_positions);
// Row-major: for each output row, compute all positions
for i in 0..rows {
    for p in 0..num_positions {
        let x_pos = &x[p * cols..(p + 1) * cols];
        let mut sum = b[i];
        for j in 0..cols {
            sum += w[i * cols + j] * x_pos[j];
        }
        y.push(sum);
    }
}
```

Output layout: `[row0_pos0, row0_pos1, ..., row1_pos0, row1_pos1, ...]`
This is row-major (rows outer, positions inner) for contiguous ConcatCollect placement.

- [ ] **Step 4: Run test, verify it passes**
- [ ] **Step 5: Verify existing LINEAR tests still pass (num_positions=1)**
- [ ] **Step 6: Commit**

### Task 2: ConcatCollect with num_positions

**Files:**
- Modify: `crates/mesh_runtime/src/pe.rs` (ConcatCollect, ConcatCollectForward variants)
- Modify: `crates/mesh_runtime/src/runtime.rs` (process_concat_fragment)
- Modify: `crates/mesh_runtime/src/program.rs` (serde)
- Test: `crates/mesh_runtime/src/runtime.rs`

- [ ] **Step 1: Add `num_positions: u32` field to ConcatCollect and ConcatCollectForward**

In `pe.rs`, add the field with default 1. In `program.rs`, add `#[serde(default)]` field defaulting to 0 (0 = infer as 1 for backward compat).

- [ ] **Step 2: Write failing test**

```rust
#[test]
fn concat_collect_batched() {
    // 2 tiles, total_rows=4, num_positions=2
    // Tile 0 (rows 0-1): [r0p0, r0p1, r1p0, r1p1]
    // Tile 1 (rows 2-3): [r2p0, r2p1, r3p0, r3p1]
    // After collect + transpose: [p0r0, p0r1, p0r2, p0r3, p1r0, p1r1, p1r2, p1r3]
}
```

- [ ] **Step 3: Update process_concat_fragment**

```rust
fn process_concat_fragment(..., num_positions: u32) -> Option<Vec<f32>> {
    let num_pos = if num_positions > 0 { num_positions as usize } else { 1 };
    let total_size = total_rows as usize * num_pos;
    let offset = fragment_offset as usize * num_pos;
    // ... existing logic but with scaled sizes ...

    if count == num_fragments {
        let mut result = pe.remove_slot(accum_slot).unwrap();
        // Transpose from (total_rows, num_pos) to (num_pos, total_rows)
        if num_pos > 1 {
            let tr = total_rows as usize;
            let mut transposed = vec![0.0; total_size];
            for r in 0..tr {
                for p in 0..num_pos {
                    transposed[p * tr + r] = result[r * num_pos + p];
                }
            }
            result = transposed;
        }
        Some(result)
    } else {
        None
    }
}
```

- [ ] **Step 4: Update all process_concat_fragment call sites to pass num_positions**
- [ ] **Step 5: Run tests, verify batched test passes and existing tests still pass**
- [ ] **Step 6: Commit**

### Task 3: ConcatCollectForward scatter mode

**Files:**
- Modify: `crates/mesh_runtime/src/pe.rs` (add `scatter: bool` to ConcatCollectForward)
- Modify: `crates/mesh_runtime/src/runtime.rs` (scatter logic)
- Modify: `crates/mesh_runtime/src/program.rs` (serde)
- Test: `crates/mesh_runtime/src/runtime.rs`

- [ ] **Step 1: Add `scatter: bool` field (default false) to ConcatCollectForward**

When scatter=true and num_positions > 1, instead of broadcasting the full result to all destinations, send row i to destination i:
```rust
if scatter {
    let row_size = result.len() / route_dests.len();
    for (i, (dest, hops)) in route_dests.iter().enumerate() {
        let row = result[i * row_size..(i + 1) * row_size].to_vec();
        // send row to dest
    }
} else {
    // existing broadcast: send full result to all dests
}
```

- [ ] **Step 2: Write test for scatter**
- [ ] **Step 3: Implement scatter in ConcatCollectForward**
- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

### Task 4: Python compiler support for num_positions

**Files:**
- Modify: `python/meshflow/compiler/schedule_ir.py` (add num_positions to ConcatCollect entries)
- Modify: `python/meshflow/compiler/artifact.py` (add num_positions, scatter fields)
- Modify: `python/meshflow/compiler/passes/lower.py` (pass through new fields)
- Modify: `python/meshflow/compiler/passes/route.py` (set num_positions from seq_len context)
- Test: `tests/python/compiler/test_artifact.py` (round-trip)

- [ ] **Step 1: Add `num_positions: int = 1` to ConcatCollectEntry and ConcatCollectForwardEntry**
- [ ] **Step 2: Add `scatter: bool = False` to ConcatCollectForwardEntry**
- [ ] **Step 3: Add matching fields to artifact task types**
- [ ] **Step 4: Update lower.py to pass through fields**
- [ ] **Step 5: Add serialization round-trip test**
- [ ] **Step 6: Run all Python tests**
- [ ] **Step 7: Commit**

---

## Chunk 2: Multi-position RMSNorm

### Task 5: RMSNorm slice parameters

**Files:**
- Modify: `crates/mesh_runtime/src/pe.rs` (add slice_offset, slice_size to RmsNormPartialSum, RmsNormNormalize)
- Modify: `crates/mesh_runtime/src/program.rs` (serde)
- Modify: `python/meshflow/compiler/schedule_ir.py`
- Modify: `python/meshflow/compiler/artifact.py`
- Modify: `python/meshflow/compiler/passes/lower.py`
- Modify: `python/meshflow/compiler/passes/route.py`

- [ ] **Step 1: Add `slice_offset: u32` and `slice_size: u32` to RmsNormPartialSum and RmsNormNormalize**

Both default to 0. When slice_size=0, use the entire input (backward compat).
When slice_size>0, extract `input[slice_offset..slice_offset+slice_size]` per position.

- [ ] **Step 2: Update Python schedule_ir and artifact types**
- [ ] **Step 3: Update lower.py and route.py to emit slice params from PlacedRmsNormTileData**
- [ ] **Step 4: Run all tests (backward compat)**
- [ ] **Step 5: Commit**

### Task 6: Per-position RMSNorm execution

**Files:**
- Modify: `crates/mesh_runtime/src/runtime.rs` (RmsNormPartialSum, RmsNormReduce, RmsNormNormalize)
- Test: `crates/mesh_runtime/src/runtime.rs`

- [ ] **Step 1: Write failing test for multi-position RMSNorm**

```rust
#[test]
fn rmsnorm_two_positions() {
    // 1 tile PE (full features), 1 reduce PE
    // Input: 2 positions of 2 features = [3, 4, 1, 0]
    // Position 0: [3, 4], sum_sq=25, scale=1/sqrt(25/2+eps)
    // Position 1: [1, 0], sum_sq=1, scale=1/sqrt(1/2+eps)
    // gamma = [1, 1]
    // Output: [3*s0, 4*s0, 1*s1, 0*s1]
}
```

- [ ] **Step 2: Update RmsNormPartialSum to compute per-position partial sums**

```rust
// Input is position-major: [p0_f0, p0_f1, ..., p1_f0, p1_f1, ...]
// Each tile handles a feature slice. With slice_offset/size, extract per position.
let full_features = if slice_size > 0 { slice_size as usize } else { data.len() };
// Infer num_positions: if slice_size > 0, we know per-position feature count
// The full input has all positions * all features. This tile's slice for position p
// starts at p * total_features + slice_offset.
// But the tile PE receives the FULL input (broadcast from upstream).
// num_positions = data.len() / total_features_per_position
// We need to know total_features_per_position... from the reduce PE's feature_count.

// Alternative: infer from data. If slice_size > 0:
//   total_features = inferred from context (feature_count from reduce)
//   num_positions = data.len() / total_features
//   For each position, extract [p*total + offset .. p*total + offset + size]

// Send per-position partial sums as a vector
let partial_sums: Vec<f32> = (0..num_positions)
    .map(|p| {
        let start = p * total_features + slice_offset;
        let slice = &data[start..start + slice_size];
        slice.iter().map(|x| x * x).sum::<f32>()
    })
    .collect();
```

The partial sum payload is now `num_positions` floats instead of 1.

- [ ] **Step 3: Update RmsNormReduce for per-position accumulation**

The accumulator stores a vector of `num_positions` running sums. On each partial sum arrival, add element-wise. On completion, compute per-position scales:

```rust
let scales: Vec<f32> = (0..num_positions)
    .map(|p| {
        let mean_sq = running_sums[p] / feature_count as f32;
        1.0 / (mean_sq + eps).sqrt()
    })
    .collect();
// Broadcast scales vector to all tile PEs
```

- [ ] **Step 4: Update RmsNormNormalize for per-position normalization**

```rust
// scale is now a vector: [scale_p0, scale_p1, ...]
// For each position p, normalize: x_slice * scale[p] * gamma
let mut result = Vec::new();
for p in 0..num_positions {
    let start = p * total_features + slice_offset;
    let x_slice = &data[start..start + slice_size];
    let s = scale[p];  // per-position scale
    for (x, g) in x_slice.iter().zip(gamma.iter()) {
        result.push(x * s * g);
    }
}
```

- [ ] **Step 5: Run tests**
- [ ] **Step 6: Verify existing single-position RMSNorm tests still pass**
- [ ] **Step 7: Commit**

### Task 7: RMSNorm collect PE

**Files:**
- Modify: `python/meshflow/compiler/passes/expand.py` (add collect PE to RmsNormGroup)
- Modify: `python/meshflow/compiler/passes/place.py` (place collect PE)
- Modify: `python/meshflow/compiler/expanded_ir.py` (update NodeExpansion output)
- Test: existing expand/place tests

The RMSNorm normalized fragments need to be gathered into a full vector for downstream consumption. Add a collect PE (reusing the existing ConcatCollectForward pattern) after the tile PEs.

- [ ] **Step 1: Update expand.py: RMSNorm output_pe_ids = [collect_id] instead of tile_ids**
- [ ] **Step 2: Update place.py: add collect PE at top of RMSNorm column, add tile→collect edges**
- [ ] **Step 3: Update route.py: generate ConcatCollectForward tasks for RMSNorm collect**
- [ ] **Step 4: Update tests**
- [ ] **Step 5: Commit**

---

## Chunk 3: Attention MATMUL redesign

### Task 8: Redesign MATMUL as matrix-vector multiply

**Files:**
- Modify: `crates/mesh_runtime/src/pe.rs` (replace MatMul variant)
- Modify: `crates/mesh_runtime/src/runtime.rs` (new execution logic)
- Modify: `crates/mesh_runtime/src/program.rs` (serde)
- Test: `crates/mesh_runtime/src/runtime.rs`

- [ ] **Step 1: Redefine MatMul TaskKind**

```rust
MatMul {
    matrix_slot: SlotId,
    vector_slot: SlotId,
    rows: u32,        // matrix rows
    cols: u32,        // matrix cols (= vector length when !transpose)
    transpose: bool,  // false: M @ v → (rows,), true: M^T @ v → (cols,)
    output_slot: SlotId,
    output_dests: Vec<(Coord, Vec<Direction>)>,
    payload_slots: Vec<SlotId>,
}
```

- [ ] **Step 2: Write tests**

```rust
#[test]
fn matmul_matrix_vector() {
    // M = [[1,2],[3,4],[5,6]] (3x2), v = [1,1]
    // M @ v = [3, 7, 11]
}

#[test]
fn matmul_transpose() {
    // M = [[1,2],[3,4]] (2x2), v = [1, 0]
    // M^T @ v = [1, 2] (first row of M)
}

#[test]
fn matmul_two_input_trigger() {
    // Both matrix and vector arrive via messages
    // Task triggers on both slots, has_slot guard
}
```

- [ ] **Step 3: Implement MatMul execution**

```rust
TaskKind::MatMul { matrix_slot, vector_slot, rows, cols, transpose, ... } => {
    let pe = self.mesh.pe_mut(coord);
    pe.counters.tasks_executed += 1;
    self.profile.total_tasks_executed += 1;

    if !pe.has_slot(matrix_slot) || !pe.has_slot(vector_slot) {
        return;
    }

    let matrix = pe.read_slot(matrix_slot).clone();
    let vector = pe.read_slot(vector_slot).clone();
    let r = rows as usize;
    let c = cols as usize;

    let result = if !transpose {
        // M @ v: result is (rows,)
        (0..r).map(|i| {
            (0..c).map(|j| matrix[i * c + j] * vector[j]).sum::<f32>()
        }).collect()
    } else {
        // M^T @ v: result is (cols,)
        (0..c).map(|j| {
            (0..r).map(|i| matrix[i * c + j] * vector[i]).sum::<f32>()
        }).collect()
    };

    // write + route
}
```

- [ ] **Step 4: Update serde in program.rs**
- [ ] **Step 5: Run tests**
- [ ] **Step 6: Commit**

### Task 9: Python compiler support for new MATMUL

**Files:**
- Modify: `python/meshflow/compiler/schedule_ir.py` (update MatMulEntry)
- Modify: `python/meshflow/compiler/artifact.py` (update MatMulTask)
- Modify: `python/meshflow/compiler/passes/lower.py`
- Modify: `python/meshflow/compiler/passes/route.py` (attention PE routing)
- Test: various

- [ ] **Step 1: Update MatMulEntry**

```python
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
    output_dests: list[...] = field(default_factory=list)
    payload_slots: list[int] = field(default_factory=list)
```

- [ ] **Step 2: Update MatMulTask in artifact.py**
- [ ] **Step 3: Update lower.py**
- [ ] **Step 4: Update route.py attention PE routing**

For each attention PE (seq_len PEs total, PE index i):
- Q collect scatters Q_row_i (d_model) → slot 0
- K collect broadcasts K matrix (seq_len * d_model) → slot 1
- V collect broadcasts V matrix (seq_len * d_model) → slot 2

Tasks on attention PE:
```
MatMul QK^T: matrix_slot=1(K), vector_slot=0(Q), rows=seq_len, cols=d_model,
             transpose=false → scores(seq_len) → slot 3
Softmax:     input_slot=3, output_slot=4
MatMul AV:   matrix_slot=2(V), vector_slot=4(softmax), rows=seq_len, cols=d_model,
             transpose=true → output(d_model) → route downstream
```

Two TaskConfig entries per MatMul (trigger on each input slot). has_slot guard computes on second arrival.

- [ ] **Step 5: Update route tests**
- [ ] **Step 6: Run all tests**
- [ ] **Step 7: Commit**

---

## Chunk 4: Integration verification

### Task 10: End-to-end batched LINEAR test

**Files:**
- Test: `tests/python/runtime/test_end_to_end.py`

- [ ] **Step 1: Write test: compile a single LINEAR with multi-position input**

```python
def test_linear_multi_position():
    # LINEAR(in=2, out=2) with identity weights
    # Input: 2 positions = [1, 2, 3, 4]
    # Expected output: [1, 2, 3, 4]
    graph = GraphIR(nodes=[
        Node(id="l", op=OpType.LINEAR, attrs={"in_features": 2, "out_features": 2}),
        Node(id="c", op=OpType.COLLECT),
    ], edges=[Edge(src_node="l", src_slot=0, dst_node="c", dst_slot=0)])
    weights = {"l": {"weight": np.eye(2), "bias": np.zeros(2)}}
    program = compile(graph, weights=weights)
    result = run_program(serialize(program), {"l": [1.0, 2.0, 3.0, 4.0]})
    # Should get [1, 2, 3, 4] back (position-major after transpose)
```

- [ ] **Step 2: Run test, verify it passes end-to-end**
- [ ] **Step 3: Commit**

### Task 11: End-to-end batched RMSNorm test

- [ ] **Step 1: Write test: compile RMSNorm with multi-position input, validate against torch**
- [ ] **Step 2: Run test**
- [ ] **Step 3: Commit**

### Task 12: End-to-end attention MatMul test

- [ ] **Step 1: Write test: Q/K/V projections → MATMUL → Softmax → MATMUL chain**
- [ ] **Step 2: Run test**
- [ ] **Step 3: Commit**
