# Milestone 7: Simple Transformer Block

## Goal

Compile and execute a single-head transformer block on the spatial mesh. Introduces four new operators (MATMUL, SOFTMAX, RMSNORM, ADD), a two-phase reduce-broadcast primitive, and broadcast routing. Validates end-to-end numerical correctness against a PyTorch reference.

## Block Structure

```
                              ┌─→ Q Proj ─→ MatMul(QK^T) ─→ Softmax ─→ MatMul(AV) ─→ Out Proj ─┐
                              │                  ↑                          ↑                  │
                    RMSNorm1 ─├─→ K Proj ────────┘                          │                  │
                    ↑         └─→ V Proj ───────────────────────────────────┘                  │
Input ─→ FORWARD ───┤                                                                          ↓
                    └──────────────────────────── (skip) ──────────────────────────────────→ Add1
                                                                                               │
                    ┌─────────────────────────── (skip) ──────────────────────────────→ Add2 ←─┘
                    │                                                                    │
                    └──→ RMSNorm2 ─→ FFN1 ─→ ReLU ─→ FFN2 ───────────────────────────────┘
                                                                                         │
                                                                                      Collect
```

Residual connections wrap both sublayers: attention output + input → Add1, FFN output + Add1 → Add2.

## Reference Dimensions

- Test case: `seq_len=4, d_model=8, d_ff=16`
- System supports arbitrary dimensions; tests use small values for speed and debuggability.

---

## Input Flow

The transformer block input enters through a FORWARD node that fans out to four destinations:

1. **RMSNorm1 tile PEs** — main path into the attention sublayer.
2. **K projection tile PEs** — key input (same as query input for self-attention).
3. **V projection tile PEs** — value input (same as query input for self-attention).
4. **Add1 PEs (skip connection)** — residual path that bypasses the attention sublayer.

The compiler generates broadcast routes from the FORWARD node to all destinations. For self-attention, Q/K/V all receive the same input. The skip connection arrives at the Add1 PEs early and waits in SRAM until the main path completes.

**SRAM pressure from skip connections**: Skip-connection data occupies SRAM until the main path catches up. For seq_len=4, d_model=8, this is 32 floats (128 bytes) per tile PE — negligible. At larger dimensions, this pressure should be monitored via the existing profiling counters.

---

## New Operators

### RMSNORM

Normalizes over the feature dimension using root mean square: `x / sqrt(mean(x^2) + eps) * gamma`.

**Graph IR**: `OpType.RMSNORM`, attrs `{eps: float}`. Learned `gamma` weights provided as operator weights. Validation: `eps` attr required, gamma weights required.

**Spatial execution (two-phase reduce-broadcast)**:

1. **Phase 1 — partial sums**: Each tile PE computes `sum(x^2)` for its local feature slice. Sends the partial sum (a single float) to the designated reduce PE.
2. **Reduce**: The reduce PE receives all partial sums, computes `1 / sqrt(total_sum / feature_count + eps)`, broadcasts the scale factor back to all tile PEs.
3. **Phase 2 — apply**: Each tile PE multiplies `x * scale * gamma` (gamma is resident in SRAM). Routes normalized output onward.

**Tile count**: Matches the preceding layer's output tile count so fragments arrive 1:1 without reshuffling.

**Rust TaskKind variants**:
- `RmsNormReduce` — reduce PE. Uses counter-based accumulation (same pattern as ConcatCollect): triggers on every partial sum arrival, increments an internal counter, only computes when counter reaches `num_tiles`. Computes scale factor and broadcasts to all tile PEs.
- `RmsNormPartialSum` — tile PEs, phase 1. Triggered when activation arrives (slot 0). Computes `sum(x^2)` for the local slice, sends partial sum to reduce PE. Completes normally.
- `RmsNormNormalize` — tile PEs, phase 2. Triggered when scale factor arrives (slot 1). Reads activation from slot 0 (still resident), scale factor from slot 1, gamma weights from slot 2. Computes `x * scale * gamma`, routes normalized output onward.

**Two separate TaskConfig entries on each tile PE**:
- TaskConfig A: `kind=RmsNormPartialSum`, `trigger_slot=0`.
- TaskConfig B: `kind=RmsNormNormalize`, `trigger_slot=1`.

**SRAM slot layout for RmsNorm tile PEs**:
- Slot 0: activation input (arrives from upstream, persists through both phases)
- Slot 1: scale factor (arrives from reduce PE after phase 1)
- Slot 2: gamma weights (pre-loaded, resident)

**Two-phase mechanism**: This uses two separate TaskConfig entries with two separate TaskKind variants, not a single task that fires twice. The runtime's existing slot-trigger model handles this naturally — TaskConfig A fires when slot 0 is written, TaskConfig B fires when slot 1 is written. No new scheduling primitive needed. The key detail: phase 1 does not consume the activation from slot 0 — it stays in SRAM for phase 2 to read.

**RmsNormReduce trigger mechanism**: The reduce PE uses the same counter-based accumulation pattern as ConcatCollect. Each partial sum message writes to a distinct slot (0..N-1) and increments a `received_fragments` counter on the PE. The RmsNormReduce task triggers on every slot write but only computes when the counter equals `num_tiles`. This matches the existing `process_concat_fragment` pattern in the runtime.

### SOFTMAX

Row-wise numerically stable softmax: `exp(x - max(x)) / sum(exp(x - max(x)))`.

**Graph IR**: `OpType.SOFTMAX`, no attrs. Validation: exactly one incoming edge (from QK^T MatMul).

**Spatial execution**: Co-located on attention PEs (no additional PEs allocated). Unlike ReLU fusion (which is a field on another task), Softmax has its own TaskKind and its own TaskConfig entry on the attention PE. It is co-located, not fused.

**Rust TaskKind**: `Softmax` — reads score row from SRAM slot, computes stable softmax, writes result to a separate output slot. The AV MatMul task watches this output slot as one of its trigger conditions.

**SRAM slot layout on attention PEs** (see MatMul section for full layout):
- Softmax reads from the QK^T result slot (slot `seq_len + 1`).
- Softmax writes to the softmax output slot (slot `seq_len + 2`).

### MATMUL

General local matrix-vector or dot-product computation.

**Graph IR**: `OpType.MATMUL`, no attrs. Validation: connectivity depends on role (QK^T vs AV), validated by the model helper's graph construction. Used for both QK^T and attention_weights @ V.

**Spatial execution (row-parallel attention)**: `seq_len` PEs, each handling one query position.
- **QK^T**: PE holds one row of Q (from Q projection). Receives all K vectors via broadcast. Computes `seq_len` dot products, producing one row of the attention score matrix.
- **AV**: PE holds its softmax output row. Receives all V vectors via broadcast. Computes weighted sum, producing one row of the attention output.

**Broadcast routing**: K and V each originate from their projection's collect PEs and need to reach all `seq_len` attention PEs. Implemented as N separate point-to-point routes from the same source — one hop list per destination. Same mechanism as ConcatCollectForward's multi-destination emission.

**SRAM slot layout on attention PEs** (seq_len=4 example):
- Slot 0: Q row (from Q projection collect, one row per attention PE)
- Slots 1..4: K vectors (slots 1, 2, 3, 4 — one per sequence position, broadcast from K projection)
- Slots 5..8: V vectors (slots 5, 6, 7, 8 — one per sequence position, broadcast from V projection)
- Slot `2*seq_len + 1` (=9 for seq_len=4): QK^T result row (written by QK^T MatMul)
- Slot `2*seq_len + 2` (=10 for seq_len=4): Softmax output row (written by Softmax)
- Slot `2*seq_len + 3` (=11 for seq_len=4): AV result row (written by AV MatMul, routed onward)

**Task triggering chain on each attention PE**:
1. **QK^T MatMul** (TaskConfig): `trigger_slot` = last K slot (slot `seq_len`). Uses ConcatCollect-like counter to detect when all K slots are populated. Q row in slot 0 is pre-populated from Q projection. Reads slots 0..seq_len, computes dot products, writes score row to slot `2*seq_len + 1`.
2. **Softmax** (TaskConfig): `trigger_slot = 2*seq_len + 1`. Reads score row, computes stable softmax, writes result to slot `2*seq_len + 2`.
3. **AV MatMul** (TaskConfig): `trigger_slot = 2*seq_len + 2`. By this point V slots (slots `seq_len+1..2*seq_len`) are already populated (V vectors arrive via broadcast in parallel with the QK^T computation). Reads softmax output + V slots, computes weighted sum, writes to slot `2*seq_len + 3`, routes result onward.

**Note on V arrival timing**: V vectors arrive independently of the QK^T → Softmax chain. They may arrive before or after Softmax completes. The AV MatMul only triggers on the Softmax output slot, so V vectors simply wait in their slots until AV fires. If V arrives after Softmax, the ordering works naturally through the event queue.

### ADD (Residual)

Element-wise addition of two activation tensors.

**Graph IR**: `OpType.ADD`, no attrs. Validation: exactly two incoming edges. Two inputs (main path + skip connection), one output.

**Spatial execution**: ADD runs on its own set of tile PEs (one per feature slice, matching the tile count of adjacent layers). Each PE receives two inputs in separate SRAM slots, adds them element-wise, and routes the result onward.

**Rust TaskKind**: `Add` — reads slot 0 (main path) and slot 1 (skip connection), writes element-wise sum to output slot, routes result onward.

**SRAM slot layout for ADD PEs**:
- Slot 0: main path input (e.g., attention output from output projection collect)
- Slot 1: skip-connection input (e.g., original input routed around the sublayer)

**Trigger**: Uses slot-based readiness. The ADD task triggers when slot 0 is written (the later-arriving input). The skip connection in slot 1 typically arrives first and waits. If the main path arrives first, the same mechanism works — the task triggers when the second slot is written. The compiler sets `trigger_slot` to whichever input arrives last based on the dataflow graph depth.

**Co-location note**: ADD PEs are distinct PEs in the placement, not co-located on other operator PEs. This keeps slot assignments simple and avoids conflicts with the consumer's own slots. The cost is a small number of additional PEs (equal to the tile count) with minimal SRAM usage.

---

## Data Flow: Attention Output → Add1 → RMSNorm2

The attention sublayer output reaches the first residual add through the following path:

1. **Output projection (LINEAR)**: Tiled like other linear layers. Each tile PE computes its fragment of the output vector.
2. **Output projection collect (ConcatCollect/ConcatCollectForward)**: Existing mechanism. Collect PE(s) accumulate fragments from output projection tiles. Once complete, routes the collected output to Add1 PEs.
3. **Add1 PEs**: Receive the attention output in slot 0, skip-connection (original input) in slot 1. Compute element-wise sum. Route result to both:
   - RMSNorm2 tile PEs (main path into FFN sublayer)
   - Add2 PEs (skip connection around FFN sublayer)
4. **RMSNorm2 → FFN1 → ReLU → FFN2**: Standard pipeline using existing LINEAR + ReLU infrastructure.
5. **FFN2 collect → Add2 PEs**: FFN output collected, routed to Add2 PEs. Add2 receives FFN output + Add1 output, sums them, routes to Collect.

---

## Compiler Pass Changes

### expand.py

Biggest changes. Currently handles LINEAR → TiledComputeGroup with optional fused activation. New expansion logic per operator:

- **RMSNORM**: Creates a tile group (one PE per feature slice) + one reduce PE. Wires two-phase connections: tile PEs → reduce PE (partial sums), reduce PE → tile PEs (scale factor). Each tile PE gets two TaskConfig entries (phase 1 and phase 2).
- **SOFTMAX**: Co-located on attention PEs. Adds a Softmax TaskConfig entry to each attention PE, positioned between QK^T and AV in the trigger chain. No additional PEs allocated.
- **MATMUL**: Creates `seq_len` attention PEs. Wires broadcast edges from K/V sources to all attention PEs. Two MATMUL TaskConfig entries on each attention PE (QK^T and AV), chained through the Softmax TaskConfig.
- **ADD**: Creates `tile_count` ADD PEs. Wires main path and skip-connection inputs to separate slots. Routes ADD output to downstream consumer.

### place.py

Currently uses sequential column-based placement. Changes:

- Attention PEs get their own column (`seq_len` PEs arranged vertically, like LINEAR tiles).
- RMSNorm tile PEs arranged vertically in a column; reduce PE placed adjacent to minimize hops.
- ADD PEs arranged vertically in a column, adjacent to their consumer.
- Mesh may need to be wider for transformer blocks (more columns than MLP alone). Mesh dimensions are derived from the placement result.

### route.py

Currently generates single point-to-point hop lists. Changes:

- **Broadcast routes**: One source PE, multiple destination PEs. Implemented as N separate point-to-point routes sharing the same source. No tree-broadcast primitive needed.
- **Skip-connection routes**: Residual input routed in parallel with the main path, arriving at ADD PEs. Just additional route entries.
- Existing XY routing works for all new patterns.

### lower.py

Mechanical translation as today. New task types map directly to new TaskKind variants. Each new op needs a lowering case that emits the right task config with correct slot assignments and routes. Slot numbering follows the layouts defined in this spec.

### schedule_ir.py

New TaskEntry variants for the new task kinds. Same structure as today: task kind, SRAM slot references, output routes.

### graph_ir.py — validation updates

- **RMSNORM**: Requires `eps` and `feature_count` attrs. Weights must include `gamma`.
- **ADD**: Requires exactly two incoming edges. `num_tiles` attr set by model helper.
- **SOFTMAX**: Requires exactly one incoming edge.
- **MATMUL**: Requires `seq_len` attr. Specific connectivity validated by the model helper's graph construction rather than generic validation (since QK^T and AV have different input patterns).

### No new IR stages

The existing pipeline stays the same: `GraphIR → Expand → SpatialIR → Route → ScheduleIR → Lower → RuntimeProgram`. Each pass just learns about more operator types.

---

## Runtime Changes (Rust)

### New TaskKind variants (pe.rs)

Six new variants added to the `TaskKind` enum:

| Variant | Triggers on | Computes | Emits |
|---------|------------|----------|-------|
| `RmsNormReduce` | Each partial sum slot (counter-based, fires when all N arrive) | `1/sqrt(sum/count + eps)` | Scale factor broadcast to N tile PEs |
| `RmsNormPartialSum` | Slot 0 (activation) | `sum(x^2)` for local slice | Partial sum to reduce PE |
| `RmsNormNormalize` | Slot 1 (scale factor) | `x * scale * gamma` (slots 0, 1, 2) | Normalized output |
| `Softmax` | Score row slot | `exp(x-max)/sum(exp(x-max))` | Result to separate output slot (local) |
| `MatMul` | Last operand slot written | Dot products or vector-matrix | Result row (routed) |
| `Add` | Later-arriving input slot | Element-wise `a + b` (slots 0, 1) | Sum (routed) |

### Task parameters

Each new TaskKind carries explicit parameters in the Rust enum. Like existing variants (e.g., `ForwardActivation`, `Linear`), all tasks that emit messages include their routing info:

- `RmsNormReduce { num_tiles: u32, feature_count: u32, eps: f32, tile_dests: Vec<(Coord, Vec<Direction>)> }` — broadcasts scale factor to all tile PEs.
- `RmsNormPartialSum { reduce_dest: Coord, reduce_hops: Vec<Direction> }` — tile PE phase 1: sends partial sum to reduce PE.
- `RmsNormNormalize { output_dests: Vec<(Coord, Vec<Direction>)> }` — tile PE phase 2: sends normalized output onward.
- `Softmax { input_slot: u32, output_slot: u32 }` — writes to local SRAM slot only, no routing needed (co-located with MatMul).
- `MatMul { operand_slots: Vec<u32>, output_slot: u32, output_dests: Vec<(Coord, Vec<Direction>)> }` — routes result to downstream PEs.
- `Add { input_slot_a: u32, input_slot_b: u32, output_slot: u32, output_dests: Vec<(Coord, Vec<Direction>)> }` — routes sum to downstream PEs. Add1 fans out to both RMSNorm2 tiles and Add2 PEs.

**Note**: Splitting RmsNormApply into `RmsNormPartialSum` and `RmsNormNormalize` as separate TaskKind variants (rather than a single variant with a phase flag) is cleaner — each variant has exactly the parameters it needs, and the runtime match arm is unambiguous.

### Task execution (runtime.rs — process_execute)

**Add**: Read two slots, write element-wise sum, route onward. Simplest new task.

**Softmax**: Read score row from input slot, compute max-subtracted stable softmax, write result to output slot. The output slot write triggers the AV MatMul task.

**MatMul**: Read operand slots (Q row + K vectors, or softmax output + V vectors), compute dot products, write result to output slot. Route onward.

**RmsNormPartialSum**: Triggered by activation arrival in slot 0. Compute `sum(x^2)` for local slice. Emit partial sum message to reduce PE. Execution completes normally — PE then waits for phase 2's trigger (slot 1 write).

**RmsNormNormalize**: Triggered by scale factor arrival in slot 1. Read activation from slot 0 (still resident), scale factor from slot 1, gamma weights from slot 2. Compute `x * scale * gamma`. Route normalized output to downstream PEs.

**RmsNormReduce**: Counter-based accumulation (like ConcatCollect). Triggers on every partial sum slot write. Increments `received_fragments` counter. When counter equals `num_tiles`, sums all partial sums from slots 0..N-1, computes `1/sqrt(total/feature_count + eps)`, broadcasts scale factor to all tile PEs via N messages.

### Broadcast message emission

Same pattern as existing ConcatCollectForward: emit N separate messages with individual hop lists. No new routing primitive in the runtime.

### Display impl for new TaskKind variants

Extend the existing `Display` impl with serde-compatible strings: `"rms_norm_reduce"`, `"rms_norm_partial_sum"`, `"rms_norm_normalize"`, `"softmax"`, `"mat_mul"`, `"add"`. Used by profiling trace events and operator timing.

---

## Model Helper

### `python/meshflow/models/transformer.py`

```python
def transformer_block(seq_len: int, d_model: int, d_ff: int, eps: float = 1e-6) -> GraphIR:
```

Constructs the full transformer block graph with all nodes and edges. Also provides a weight generation helper:

```python
def transformer_weights(seq_len: int, d_model: int, d_ff: int) -> dict[str, Any]:
```

Returns a dict of all required weights: Q/K/V/output projection matrices and biases, FFN1/FFN2 matrices and biases, two sets of RMSNorm gamma vectors.

### `python/meshflow/models/reference.py`

Extended with a torch reference for the transformer block:

```python
def transformer_block_reference(inputs: np.ndarray, weights: dict, seq_len: int, d_model: int, d_ff: int, eps: float = 1e-6) -> np.ndarray:
```

Runs the equivalent computation in pure PyTorch, returns expected output for numerical validation.

---

## Testing Strategy

### Rust unit tests (per operator)

- **MatMul**: Dot product correctness on small vectors, vector-matrix product matches numpy.
- **Softmax**: Numerical stability with large values, output sums to 1.0, matches reference.
- **RmsNormReduce / RmsNormApply**: Partial sum accumulation, scale factor correctness, normalized output matches reference.
- **Add**: Element-wise addition correctness.

### Python end-to-end tests

- `test_transformer_block_basic`: seq_len=4, d_model=8, d_ff=16. Compile, run, compare against torch reference. Tolerance: `atol=1e-5, rtol=1e-5`.
- `test_transformer_block_identity_weights`: Identity-like weights for analytically predictable output.
- `test_rmsnorm_standalone`: Single RMSNorm node → compile → run → validate.
- `test_attention_standalone`: QKV projections + matmul + softmax + AV + output projection (no FFN/norm).
- `test_residual_passthrough`: Verify residual connections preserve skip input correctly.

### Profiling validation

Run the transformer block through M6 visualization tools. Verify:
- All new operators appear in the event timeline.
- Operator latency chart shows meaningful data for new task kinds.
- PE heatmap reflects the wider mesh usage.
- Route contention shows broadcast patterns.

---

## Out of Scope (Deferred)

- **Multi-head attention** — straightforward extension once single-head works, adds placement complexity.
- **Causal masking** — needed for autoregressive generation, not for single-block validation.
- **Dropout** — identity at inference time, skipped entirely.
- **Positional encoding** — orthogonal to mesh execution; baked into test inputs.
- **KV cache** — needed for generation, not single-pass inference.
- **Inner-dimension matmul tiling** — splitting d_model across PEs for a single dot product; not needed at test dimensions.
- **Tree-broadcast routing** — N point-to-point fan-out is sufficient for small seq_len.
- **API integration** — transformer block works through `compile()` + `run_program()`; no new API endpoint.
