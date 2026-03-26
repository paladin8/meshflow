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

The transformer block input enters through a FORWARD node that fans out to:

1. **RMSNorm1 tile PEs** — main path into the attention sublayer. RMSNorm1 normalizes the input, then its output fans out to Q/K/V projection tile PEs.
2. **Add1 PE (skip connection)** — residual path that bypasses the attention sublayer.

For self-attention, Q/K/V projections all receive the same normalized input from RMSNorm1's output. The skip connection carries the *unnormalized* original input to Add1.

**SRAM pressure from skip connections**: Skip-connection data occupies SRAM until the main path catches up. For seq_len=4, d_model=8, this is 32 floats (128 bytes) — negligible. At larger dimensions, this pressure should be monitored via the existing profiling counters.

---

## New Operators

### RMSNORM

Normalizes over the feature dimension using root mean square: `x / sqrt(mean(x^2) + eps) * gamma`.

**Graph IR**: `OpType.RMSNORM`, attrs `{eps: float, feature_count: int}`. Learned `gamma` weights provided as operator weights. Validation: `eps` and `feature_count` attrs required, gamma weights required.

**Spatial execution (two-phase reduce-broadcast)**:

1. **Phase 1 — partial sums**: Each tile PE computes `sum(x^2)` for its local feature slice (using `slice_offset` and `slice_size` to index into position-major data). For multi-position inputs, computes per-position partial sums. Sends the partial sum vector to the designated reduce PE.
2. **Reduce**: The reduce PE receives all partial sums, computes per-position `1 / sqrt(total_sum / feature_count + eps)`, broadcasts the scale factor vector back to all tile PEs.
3. **Phase 2 — apply**: Each tile PE multiplies `x * scale * gamma` per position (gamma is resident in SRAM). Outputs in row-major order (rows outer, positions inner) for ConcatCollect fragment placement. Routes normalized output onward.

**Tile count**: Matches the preceding layer's output tile count so fragments arrive 1:1 without reshuffling.

**Rust TaskKind variants**:
- `RmsNormReduce { num_tiles, feature_count, eps, tile_dests, scale_slot }` — reduce PE. Uses counter-based accumulation (same pattern as ConcatCollect): triggers on every partial sum arrival, increments an internal counter, only computes when counter reaches `num_tiles`. Computes per-position scale factor and broadcasts to all tile PEs.
- `RmsNormPartialSum { input_slot, reduce_dest, reduce_hops, partial_sum_slot, slice_offset, slice_size, feature_count }` — tile PEs, phase 1. Triggered when activation arrives (slot 0). Computes per-position `sum(x^2)` for the local slice using `slice_offset`/`slice_size` to index into position-major data. Sends partial sum vector to reduce PE.
- `RmsNormNormalize { input_slot, scale_slot, gamma_slot, output_dests, payload_slots, slice_offset, slice_size }` — tile PEs, phase 2. Triggered when scale factor arrives (slot 1). Reads activation from slot 0 (still resident), per-position scale factor from slot 1, gamma weights from slot 2. Computes `x * scale * gamma` per position, outputs in row-major order. Routes normalized output onward.

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
- Softmax reads from the QK^T result slot (slot 3).
- Softmax writes to the softmax output slot (slot 4).

### MATMUL

Matrix-vector multiply: `M @ v` (transpose=false, output has `rows` elements) or `M^T @ v` (transpose=true, output has `cols` elements).

**Graph IR**: `OpType.MATMUL`, attrs `{seq_len: int, d_model: int}`. Validation: `seq_len` attr required. Used for both QK^T and AV within attention chains.

**Spatial execution (row-parallel attention)**: `seq_len` PEs, each handling one query position. The expand pass detects MATMUL → SOFTMAX → MATMUL chains and co-locates all three operations on the same set of attention PEs.

- **QK^T**: `K @ Q` where K is the full key matrix (seq_len × d_model) broadcast to all attention PEs, and Q is a single row scattered per PE. `transpose=false` produces a score vector of length `seq_len`.
- **AV**: `V^T @ softmax_weights` where V is the full value matrix (seq_len × d_model) broadcast to all attention PEs. `transpose=true` produces an output vector of length `d_model`.

**Broadcast vs scatter routing**: K and V projection collect PEs **broadcast** the full matrix to all attention PEs (each PE gets the complete K/V). Q projection collect PE **scatters** — row i goes to attention PE i. The route pass auto-detects scatter mode when a collect PE routes to multiple ATTENTION_PEs at Q slot (slot 0).

**SRAM slot layout on attention PEs** (fixed, independent of seq_len):
- Slot 0: Q row (`d_model` floats, scattered from Q projection collect)
- Slot 1: K matrix (`seq_len × d_model` floats, broadcast from K projection collect)
- Slot 2: V matrix (`seq_len × d_model` floats, broadcast from V projection collect)
- Slot 3: QK^T result (`seq_len` floats, written by QK^T MatMul)
- Slot 4: Softmax output (`seq_len` floats, written by Softmax)
- Slot 5: AV result (`d_model` floats, written by AV MatMul, routed onward)

**Task triggering chain on each attention PE** (dual-trigger with has_slot guard):
1. **QK^T MatMul**: Two TaskConfig entries with `trigger_slot=0` (Q) and `trigger_slot=1` (K). Both fire, but a `has_slot` guard ensures computation only runs when both Q and K are present. Both input slots are consumed after computation to prevent re-triggering. Writes score vector to slot 3.
2. **Softmax**: Single TaskConfig with `trigger_slot=3`. Reads QK^T scores, computes stable softmax, writes result to slot 4. The `task_write_slot` helper ensures co-located tasks (AV MatMul) are properly triggered.
3. **AV MatMul**: Two TaskConfig entries with `trigger_slot=2` (V) and `trigger_slot=4` (softmax output). Same has_slot guard pattern — computes only when both V and softmax output are present. Both consumed after computation. Writes to slot 5, routes result onward.

**Note on V arrival timing**: V vectors arrive independently of the QK^T → Softmax chain. They may arrive before or after Softmax completes. The AV MatMul has_slot guard handles both orderings naturally.

**Attention output gathering** (seq_len > 1): When seq_len > 1, each attention PE produces one row of the output. An attention collect PE (ConcatCollect) gathers these rows before routing to the next operator. For seq_len = 1, the single attention PE routes directly downstream.

### ADD (Residual)

Element-wise addition of two activation tensors.

**Graph IR**: `OpType.ADD`, no attrs. Validation: exactly two incoming edges. Two inputs (main path + skip connection), one output.

**Spatial execution**: ADD is a **single-PE operation** — it expands as a PassthroughGroup, not a tiled group. The upstream collect PE already gathers fragments into a single vector, so there is no need to tile the addition across multiple PEs.

**Rust TaskKind**: `Add { input_slot_a, input_slot_b, output_slot, output_dests, payload_slots }` — reads two input slots, writes element-wise sum to output slot, routes result to all downstream destinations with per-destination payload slots.

**SRAM slot layout for ADD PE**:
- Slot 0: first input (whichever edge the compiler maps to dst_slot 0)
- Slot 1: second input (mapped to dst_slot 1)
- Slot 2: output (written by Add, routed onward)

**Trigger (dual-trigger with has_slot guard)**: The compiler generates **two AddEntry TaskConfig entries** — one per input slot. Both fire when their respective input arrives, but a `has_slot` guard in the runtime ensures computation only runs when both inputs are present. Both input slots are consumed after computation to prevent re-triggering. This pattern is identical to MatMul's dual-trigger approach.

**Single-PE note**: ADD PEs are distinct PEs in the placement, not co-located on other operator PEs. Unlike the original design (which proposed tile_count ADD PEs), the single-PE approach is simpler and sufficient because the upstream collect already gathers into a single vector.

---

## Data Flow: Attention Output → Add1 → RMSNorm2

The attention sublayer output reaches the first residual add through the following path:

1. **Output projection (LINEAR)**: Tiled like other linear layers. Each tile PE computes its fragment of the output vector.
2. **Output projection collect (ConcatCollect/ConcatCollectForward)**: Existing mechanism. Collect PE gathers fragments from output projection tiles. Once complete, routes the collected output to the Add1 PE.
3. **Add1 PE** (single PE): Receives the attention output in one slot and the skip-connection (original input) in another. Computes element-wise sum. Routes result to both:
   - RMSNorm2 tile PEs (main path into FFN sublayer)
   - Add2 PE (skip connection around FFN sublayer)
4. **RMSNorm2 → FFN1 → ReLU → FFN2**: Standard pipeline using existing LINEAR + ReLU infrastructure.
5. **FFN2 collect → Add2 PE**: FFN output collected, routed to Add2 PE. Add2 receives FFN output + Add1 output, sums them, routes to Collect.

---

## Compiler Pass Changes

### expand.py

Biggest changes. Handles LINEAR → TiledComputeGroup with optional fused activation. New expansion logic per operator:

- **RMSNORM**: Creates an `RmsNormGroup` (tile PEs + reduce PE + collect PE). Uses `reserved_rows=2` for reduce + collect when computing tile count. `NodeExpansion` maps input_pe_ids to tile PEs and output_pe_ids to the collect PE.
- **SOFTMAX + MATMUL (attention chains)**: The `_detect_attention_chains()` function identifies MATMUL → SOFTMAX → MATMUL patterns. The first MATMUL creates an `AttentionGroup` with `seq_len` PEs. SOFTMAX and AV MATMUL are marked as fused (added to `fused_nodes` set, skipped in the main loop). For seq_len > 1, an attention collect PE is added to output_pe_ids for gathering per-position outputs.
- **ADD**: Expands as a `PassthroughGroup` (single PE). No tiling — the upstream collect already gathers into a single vector.
- **Standalone SOFTMAX**: Also a `PassthroughGroup` (single PE).

### expanded_ir.py

Group types: `TiledComputeGroup`, `RmsNormGroup`, `AttentionGroup`, `PassthroughGroup`. Each group tracks an `origin_id` back to the GraphIR node. `NodeExpansion` tracks `input_pe_ids` and `output_pe_ids` for inter-group edge generation.

### place.py

Column-per-group layout for mixed graphs (passthrough-only graphs use row-major). Changes:

- Attention PEs get their own column (`seq_len` PEs arranged vertically). For seq_len > 1, an attention collect PE is placed below.
- RMSNorm: tile PEs vertically in a column, reduce PE and collect PE placed below.
- ADD PE gets its own column (single PE).
- Internal edges (tile→reduce, reduce→tile, tile→collect, attn→collect) are generated within the placement pass.
- Inter-group edges are generated from original GraphIR edges mapped through `NodeExpansion`.
- `PlacedNodeKind` enum provides clean dispatch: FORWARD, COLLECT_SIMPLE, LINEAR_TILE, LINEAR_COLLECT, RMSNORM_TILE, RMSNORM_REDUCE, ATTENTION_PE, ADD, SOFTMAX.
- Mesh dimensions are derived from the placement result.

### route.py

Dispatches on `PlacedNodeKind` for each node. Key routing patterns:

- **LINEAR_COLLECT**: Detects scatter mode when routing to multiple ATTENTION_PEs at Q slot (slot 0). Uses `ConcatCollectForward` with `scatter=True` for Q distribution, `scatter=False` (broadcast) for K/V.
- **ATTENTION_PE**: Generates co-located tasks — QK^T MatMul (dual trigger on Q/K slots), Softmax (trigger on QK^T output), AV MatMul (dual trigger on V/softmax output). Fixed 6-slot SRAM layout.
- **ADD**: Generates dual AddEntry tasks (one per input slot) with has_slot guard pattern. Output slot = max(input slots) + 1.
- **RMSNORM_TILE**: Phase 1 (PartialSum with slice params) + Phase 2 (Normalize with output routing). Gamma weights loaded into SRAM slot 2.
- **RMSNORM_REDUCE**: Counter-based accumulation entries (one per partial sum slot).
- **Broadcast routes**: One source, multiple destinations. N separate point-to-point XY routes.
- **ConcatCollectForward `fragment_rows`**: Set from the base/remainder tile distribution for exact `num_positions` inference in the runtime.
- **`payload_slots`**: Per-destination slot assignment. Used by ConcatCollectForward, Add, MatMul, and RmsNormNormalize to deliver data to the correct SRAM slot on each destination PE.

### lower.py

Mechanical translation as today. New task types map directly to new TaskKind variants. Each new op needs a lowering case that emits the right task config with correct slot assignments and routes. Slot numbering follows the layouts defined in this spec.

### schedule_ir.py

New TaskEntry variants for the new task kinds. Same structure as today: task kind, SRAM slot references, output routes.

### graph_ir.py — validation updates

- **RMSNORM**: Requires `eps` and `feature_count` attrs. Weights must include `gamma`.
- **ADD**: Requires exactly two incoming edges. No special attrs needed.
- **SOFTMAX**: Requires exactly one incoming edge.
- **MATMUL**: Requires `seq_len` attr. `d_model` attr optional (used by attention PE routing for SRAM layout). Specific connectivity validated by the model helper's graph construction.

### No new IR stages

The existing pipeline stays the same: `GraphIR → Expand → SpatialIR → Route → ScheduleIR → Lower → RuntimeProgram`. Each pass just learns about more operator types.

---

## Runtime Changes (Rust)

### TaskKind variants (pe.rs)

Six new variants added to the `TaskKind` enum (11 total including existing ForwardActivation, CollectOutput, Linear, ConcatCollect, ConcatCollectForward):

| Variant | Trigger pattern | Computes | Emits |
|---------|----------------|----------|-------|
| `RmsNormReduce` | Counter-based (fires when all N partial sums arrive) | Per-position `1/sqrt(sum/count + eps)` | Scale factor vector broadcast to N tile PEs |
| `RmsNormPartialSum` | Slot 0 (activation) | Per-position `sum(x^2)` for local slice | Partial sum vector to reduce PE |
| `RmsNormNormalize` | Slot 1 (scale factor) | Per-position `x * scale * gamma` (slots 0, 1, 2) | Normalized output (row-major) |
| `Softmax` | QK^T result slot | `exp(x-max)/sum(exp(x-max))` | Result to local output slot |
| `MatMul` | Dual trigger + has_slot guard | `M @ v` or `M^T @ v` | Result vector (routed) |
| `Add` | Dual trigger + has_slot guard | Element-wise `a + b` | Sum (routed) |

### Task parameters (pe.rs)

Each TaskKind carries explicit parameters. Full parameter lists:

- `RmsNormReduce { num_tiles, feature_count, eps, tile_dests, scale_slot }` — broadcasts per-position scale factor to all tile PEs.
- `RmsNormPartialSum { input_slot, reduce_dest, reduce_hops, partial_sum_slot, slice_offset, slice_size, feature_count }` — phase 1: per-position partial sums using slice params.
- `RmsNormNormalize { input_slot, scale_slot, gamma_slot, output_dests, payload_slots, slice_offset, slice_size }` — phase 2: per-position normalization, row-major output.
- `Softmax { input_slot, output_slot }` — writes to local SRAM slot only, no routing. Uses `task_write_slot` helper to trigger co-located tasks.
- `MatMul { matrix_slot, vector_slot, rows, cols, transpose, output_slot, output_dests, payload_slots }` — explicit matrix-vector multiply. `transpose=false` → output length `rows`, `transpose=true` → output length `cols`.
- `Add { input_slot_a, input_slot_b, output_slot, output_dests, payload_slots }` — routes sum to downstream PEs with per-destination payload slots.

### has_slot guard pattern

MatMul and Add use a dual-trigger pattern: two TaskConfig entries (one per input slot), both fire when their respective input arrives. A `has_slot` check ensures computation only runs when both inputs are present. After computation, both input slots are consumed (`pe.remove_slot`) to prevent double execution on re-trigger.

### Multi-position support

LINEAR, RmsNorm, and ConcatCollect all support `num_positions > 1`:
- **LINEAR**: Infers `num_positions = input.len() / tile_cols`, loops over positions, outputs row-major (rows outer, positions inner).
- **ConcatCollect**: Uses `fragment_rows` field for exact `num_positions = fragment.len() / fragment_rows` inference. Transposes completed buffer from row-major to position-major on completion.
- **RmsNormPartialSum/Normalize**: Uses `slice_offset`/`slice_size` with `feature_count` to handle position-major input, output in row-major for ConcatCollect.
- **RmsNormReduce**: Per-position accumulation and per-position scale computation.

### Task execution (runtime.rs — process_execute)

**Add**: has_slot guard → read two slots → element-wise sum → consume both inputs → write output → route. Counter incremented after guard (counts completions, not triggers).

**Softmax**: Read score vector from input slot → compute max-subtracted stable softmax → write result to output slot via `task_write_slot` (triggers co-located tasks like AV MatMul).

**MatMul**: has_slot guard → read matrix and vector slots → multiply (`M @ v` or `M^T @ v`) → consume both inputs → write output → route. Counter incremented after guard.

**RmsNormPartialSum**: Triggered by activation arrival. Compute per-position `sum(x^2)` for local slice. Emit partial sum vector to reduce PE. Activation stays in SRAM for phase 2.

**RmsNormNormalize**: Triggered by scale factor arrival. Read activation (slot 0), scale (slot 1), gamma (slot 2). Compute per-position `x * scale * gamma`. Output in row-major order. Route to downstream PEs.

**RmsNormReduce**: Counter-based accumulation. Triggers on every partial sum slot write. When counter equals `num_tiles`, sums all per-position partial sums, computes per-position `1/sqrt(total/feature_count + eps)`, broadcasts scale vector to all tile PEs.

### Broadcast message emission

Same pattern as ConcatCollectForward: emit N separate messages with individual hop lists and per-destination `payload_slots`. No new routing primitive.

### Display impl

All TaskKind variants have serde-compatible Display strings: `"rms_norm_reduce"`, `"rms_norm_partial_sum"`, `"rms_norm_normalize"`, `"softmax"`, `"mat_mul"`, `"add"`. Used by profiling trace events and operator timing.

---

## Model Helper — COMPLETE (CP5)

### `python/meshflow/models/transformer.py`

```python
def transformer_block(seq_len: int, d_model: int, d_ff: int, eps: float = 1e-6) -> GraphIR:
```

Constructs the full transformer block GraphIR with 16 nodes and 19 edges:
- FORWARD node (input)
- RMSNorm1 node (pre-attention normalization)
- Q/K/V LINEAR projection nodes
- QK^T MATMUL → SOFTMAX → AV MATMUL attention chain
- Output projection LINEAR
- Add1 (residual: attention output + original input)
- RMSNorm2 (pre-FFN normalization)
- FFN1 LINEAR → ReLU → FFN2 LINEAR
- Add2 (residual: FFN output + Add1 output)
- COLLECT node (output)

Edge connectivity matches the block structure diagram. Self-attention: Q/K/V projections all receive the same input (RMSNorm1 output). Skip connections carry data from FORWARD→Add1 (dst_slot=1) and Add1→Add2 (dst_slot=1).

```python
def transformer_weights(d_model: int, d_ff: int, seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
```

Returns weights keyed by node ID with Xavier-like initialization: Q/K/V/output projection matrices (d_model × d_model) and zero biases, FFN1 matrix (d_model × d_ff) and zero bias, FFN2 matrix (d_ff × d_model) and zero bias, two RMSNorm gamma vectors (ones, d_model). The `seed` parameter controls the random number generator for reproducible tests.

### `python/meshflow/models/reference.py`

Extended with a torch reference for the transformer block:

```python
def reference_transformer_block(x: torch.Tensor, weights: dict[str, dict[str, Any]], eps: float = 1e-6) -> torch.Tensor:
```

Runs the equivalent computation using existing `reference_linear()` and `reference_rmsnorm()` plus torch attention math (`Q @ K.T → softmax → @ V`). Dimensions are inferred from tensor shapes — no redundant `seq_len`/`d_model`/`d_ff` parameters. A `_to_tensor()` helper converts numpy weight arrays to torch tensors on the fly.

---

## Testing Strategy

### Rust unit tests (per operator) — COMPLETE

91 Rust tests covering all task kinds:

- **MatMul**: Matrix-vector multiply correctness, transpose mode, has_slot guard, slot consumption.
- **Softmax**: Numerical stability with large values, output sums to 1.0, task_write_slot triggering.
- **RmsNormPartialSum / RmsNormNormalize / RmsNormReduce**: Partial sum accumulation, per-position scale factor, normalized output, slice offset/size handling.
- **Add**: Element-wise addition, has_slot guard, slot consumption, dual-trigger pattern.
- **ConcatCollect**: fragment_rows-based num_positions inference, row-major → position-major transpose.

### Python end-to-end tests — COMPLETE

Per-operator tests in `tests/python/runtime/test_num_positions.py`:
- `TestBatchedLinear`: Identity 2-position, torch-validated 2-position.
- `TestMultiPositionRmsNorm`: Single position, two positions — compared against `reference_rmsnorm()`.
- `TestAttentionMatMulChain`: seq_len=1 identity projections, seq_len=4 torch-validated.

Full transformer block tests in `tests/python/runtime/test_transformer_block.py`:
- `test_residual_passthrough`: Zero weights → output equals input. Validates skip connections preserve input through both Add1 and Add2.
- `test_basic_with_torch_validation`: seq_len=4, d_model=8, d_ff=16. Random weights, compared against `reference_transformer_block()`. Tolerance: `atol=1e-3`.
- `test_small_dimensions`: seq_len=2, d_model=4, d_ff=8. Smaller dimensions for fast debugging.
- `test_non_divisible_dimensions`: seq_len=3, d_model=7, d_ff=11, mesh_height=5. Nothing divides evenly — exercises remainder logic in tile distribution, ConcatCollect fragment handling, and RMSNorm slicing.

### Profiling validation (deferred)

The transformer block can be run through M6 visualization tools to verify new operators appear in the event timeline, operator latency chart, PE heatmap, and route contention visualization. Not yet formally validated.

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
