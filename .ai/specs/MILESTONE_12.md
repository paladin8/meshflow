# Milestone 12: Routing & Placement Optimizations

## Goal

Reduce communication overhead through placement and compiler optimizations targeting the bottlenecks exposed by M11's wormhole routing model.

**Motivation:** Profiling shows that 54% of the medium config's 6901-cycle runtime is communication overhead (routing, link contention, pipeline stalls). The two worst offenders:

1. **RMSNorm: 19× comm/compute ratio.** Only 49 cycles of compute but 945 cycles of communication. The current 5-phase protocol (partial sums → reduce → scale broadcast → normalize → collect → broadcast) forces 5 sequential communication steps for a trivial computation.

2. **FFN-down+ADD: 823 cycles of communication overhead.** Fragment gather serialization at the collect PE, tile size imbalance (769 vs 513 cycle tiles), and a 2435-cycle gap where the residual sits idle waiting for the FFN output.

Additionally, the Phase 1 stagger (already implemented) shifts some collect PEs off-center, increasing gather distance for columns where gather dominates. Centering those collects back recovers hops on the critical path.

## Key design decisions

1. **Three-way staggered collect rows (Phase 1, implemented):** Cycle the collect PE's Y-position across three positions based on `col % 3`. Distributes east-west broadcast traffic across three rows. Attention groups excluded from stagger (N-to-1 gather is sensitive to collect distance).

2. **Simplified RMSNorm (Phase 2):** Eliminate the reduce PE and distributed normalize. The collect PE receives raw data slices from tiles, computes the full RMSNorm in-place (sum of squares, scale, normalize, apply gamma), and broadcasts the result. Reduces the protocol from 5 communication phases to 2, saving ~900 cycles on the medium config's critical path.

3. **Centered collect for gather-dominated columns (Phase 3):** Exempt columns where the tile→collect gather is the critical-path bottleneck from the 3-way stagger. The collect stays at center to minimize the max tile-to-collect distance. Applies to FFN columns and any other column where compute imbalance means the collect waits for the slowest tile.

4. **Fused ConcatAdd for pipelined residual connections (Phase 4):** A new runtime task that combines fragment collection with element-wise addition. As each fragment arrives from a linear tile, it's immediately added to the corresponding rows of a pre-arrived residual vector. No assembly step — the result is ready when the last fragment arrives. Eliminates the collect→ADD pipeline stall at residual connections.

## Sequencing

Four phases:

1. **Staggered collect row placement** — Already implemented (commit `833dc56`).

2. **Simplified RMSNorm** — Eliminate reduce PE. Collect PE computes RMSNorm directly on assembled raw data. Update placement, routing, and runtime.

3. **Centered collect for gather-dominated columns** — Exempt FFN and other gather-heavy columns from stagger. Placement-only change.

4. **Fused ConcatAdd** — New runtime task. Compiler emits ConcatAdd at residual connection points (projection+ADD, FFN-down+ADD). Re-baseline benchmarks.

---

## Phase 1: Staggered Collect Row Placement

Already implemented in commit `833dc56`. See commit for details.

Results: small final_timestamp -3.5%, link_contentions -12%. Medium link_contentions -2%, link_wait_cycles -7%.

---

## Phase 2: Simplified RMSNorm

### Current behavior (5 communication phases)

```
Phase 1: tiles compute sum(x²) for local slice → route to reduce PE (1-5 hops)
Phase 2: reduce PE accumulates partial sums, computes scale = 1/√(mean + ε)
Phase 3: reduce PE broadcasts scale to all tiles (1-5 hops)
Phase 4: tiles normalize: x * scale * γ → route to collect PE (1-4 hops)
Phase 5: collect PE assembles fragments → broadcasts to next layer
```

For the medium config, RMSNorm-1 takes 994 cycles on the critical path with only 49 cycles of actual compute (19× comm/compute ratio). The reduce PE at the top of the column (row 7) is 5+ hops from the farthest tiles. The scale broadcast and normalize phases are pure communication overhead that exists only because the normalization is distributed.

### New behavior (2 communication phases)

```
Phase 1: tiles send raw data slices → route to collect PE (1-4 hops)
Phase 2: collect PE assembles full vector, computes RMSNorm in-place,
         broadcasts normalized result to next layer
```

The reduce PE is eliminated. The collect PE receives raw (unnormalized) data from each tile, assembles the full feature vector, computes the complete RMSNorm (sum of squares → scale → normalize × gamma), and broadcasts.

### Compute tradeoff

The collect PE now does the full RMSNorm computation instead of just concatenation. For d_model=16 and 8 positions:
- Sum of squares: 16 × 8 = 128 multiplies
- Scale: 8 divisions + 8 sqrt
- Normalize × gamma: 16 × 8 = 128 multiplies

Total: ~300 element operations. At cost_per_element=1, this adds ~300 cycles to the collect PE's task. But this replaces ~900 cycles of communication (phases 2-4), for a net saving of ~600 cycles.

### Placement changes

**`_place_rmsnorm_group`** (place.py): Remove the reduce PE. The group now consists of only tile PEs and one collect PE. Column height drops from `num_tiles + 2` to `num_tiles + 1`.

Internal edges change from:
- tiles → reduce (partial sums)
- reduce → tiles (scale broadcast)
- tiles → collect (normalized fragments)

To:
- tiles → collect (raw data fragments)

### Routing changes

**Route pass**: Remove `_route_rmsnorm_tile` phase 1 (partial sum routing) and phase 2 (normalize routing). Tiles now have a single task: forward raw data to the collect PE. The tile PE no longer needs RmsNormPartialSum or RmsNormNormalize tasks — it's just a ForwardActivation.

**Collect PE task**: Replace ConcatCollectForward with a new task type `RmsNormCollectForward` that:
1. Gathers fragments (reuses the existing concat accumulator)
2. When all fragments arrive, computes RMSNorm in-place on the assembled vector
3. Broadcasts the normalized result

### Runtime changes

Add `RmsNormCollectForward` variant to `TaskKind`:
```rust
TaskKind::RmsNormCollectForward {
    num_fragments: u32,
    total_rows: u32,
    fragment_offset: u32,
    fragment_rows: u32,
    num_positions: u32,
    feature_count: u32,
    eps: f32,
    gamma_slot: SlotId,
    routes: Vec<BroadcastRouteRuntime>,
}
```

Execution: reuses `process_concat_fragment` for accumulation. When the final fragment arrives:
1. Read gamma from `gamma_slot`
2. For each position: compute `scale = 1/√(sum(x²)/feature_count + eps)`
3. Apply `x[i] = x[i] * scale * gamma[i]` in-place
4. Broadcast the result via routes

Corresponding changes in `program.rs` (deserialization), `artifact.py` (dataclass), `schedule_ir.py` (entry type), `lower.py` (lowering).

### SRAM changes

The collect PE needs the gamma weights pre-loaded in SRAM. Currently gamma is on each tile PE (slot 2). With simplified RMSNorm, gamma moves to the collect PE.

Tile PEs no longer need gamma weights in SRAM. They also don't need separate RmsNormPartialSum and RmsNormNormalize tasks — they just forward their raw data slice to the collect.

### Expected improvement

Medium config:
- RMSNorm communication phases: 5 → 2 (eliminate reduce, scale broadcast, distributed normalize)
- Reduce PE: eliminated (saves 1 PE per RMSNorm group, column height -1)
- RMSNorm critical-path time: ~994 → ~400 cycles (estimated)
- Two RMSNorm stages (RMSNorm-1 + RMSNorm-2) together save ~1000+ cycles

### Testing

- Numerical correctness: RMSNorm output must match reference implementation.
- Eliminate reduce PE from placement tests.
- Verify gamma weights loaded on collect PE.
- Re-baseline benchmark thresholds.

---

## Phase 3: Centered Collect for Gather-Dominated Columns

### Motivation

Phase 1's 3-way stagger distributes broadcast traffic but can hurt gather-dominated columns by moving the collect off-center. For columns where the critical path goes through tile→collect gather (like FFN-down), the stagger adds 1 extra hop for the farthest tile, delaying the collect and everything downstream.

### Change

Exempt gather-dominated columns from the stagger. These are columns where:
- The tile compute is the bottleneck (large tile_rows × tile_cols)
- The collect waits for the slowest tile's fragment

In practice: FFN columns (cols 8, 9 in the small config) and any column with tiles wider than d_model. The heuristic: if `tile_cols > d_model` (i.e., the column processes the wider FFN dimension), keep the collect centered.

Implementation: pass `stagger_offset=1` (delta=0 → center) for these columns in `_place_tiled_compute_group`, similar to how attention groups are already exempted.

### Expected improvement

Small: 1-2 hop reduction for FFN-down critical-path tiles.
Medium: similar, recovering the stagger-induced regression on the FFN critical path.

---

## Phase 4: Fused ConcatAdd

### Current behavior

At residual connection points (Proj+ADD, FFN-down+ADD), the pipeline is:

```
Linear tiles compute fragments → route to collect PE (1-5 hops)
Collect PE gathers ALL fragments → assembles complete vector
ADD: element-wise add of (assembled vector) + (residual from earlier layer)
ADD broadcasts result to next layer
```

The collect PE must wait for every fragment before the ADD can start. In the medium config, the residual arrives at t=4077 but the FFN-down collect doesn't finish until t=6512 — a 2435-cycle gap where the residual sits idle.

### New behavior

A fused `ConcatAdd` task replaces the ConcatCollectForward + Add pair:

```
Linear tiles compute fragments → route to collect+ADD PE
ConcatAdd: as each fragment arrives, add to corresponding rows of residual
When last fragment arrives, result is complete → broadcast to next layer
```

Each fragment is added to the residual immediately upon arrival, without waiting for other fragments. The output is ready as soon as the last fragment arrives — no separate assembly or ADD step.

### Runtime changes

Add `ConcatAdd` variant to `TaskKind`:
```rust
TaskKind::ConcatAdd {
    num_fragments: u32,
    total_rows: u32,
    fragment_offset: u32,
    fragment_rows: u32,
    num_positions: u32,
    residual_slot: SlotId,
    output_slot: SlotId,
    routes: Vec<BroadcastRouteRuntime>,
}
```

Execution:
1. On each fragment arrival: read fragment from trigger_slot, read residual from residual_slot
2. Add fragment elements to corresponding rows of an accumulator (initialized from residual on first fragment)
3. When all fragments have arrived: broadcast the accumulated result via routes

This reuses the `process_concat_fragment` accumulator pattern but initializes the accumulator from the residual instead of zeros, and adds each fragment instead of copying.

### Compiler changes

The routing pass detects co-located ConcatCollectForward + Add patterns (already identified by the ADD co-location pass in place.py). Instead of emitting separate ConcatCollectForward and Add tasks, emit a single ConcatAdd task with the residual_slot pointing to the slot where the residual arrives.

### Expected improvement

Medium config:
- Eliminates the collect assembly latency (waiting for all fragments before ADD)
- The last-fragment-to-output delay drops from ~130 cycles (collect assembly + ADD compute) to ~1 cycle (ADD already done incrementally)
- Proj+ADD: save ~100-200 cycles
- FFN-down+ADD: save ~100-200 cycles

The savings are modest per-stage but compound across two residual connections.

---

## Re-baseline (after all phases)

### Updated profiling expectations

After all optimizations:
- **RMSNorm overhead dramatically reduced:** 5-phase → 2-phase protocol, ~1000+ cycles saved across both RMSNorm stages
- **FFN gather distance minimized:** Centered collect for FFN columns
- **Residual connections pipelined:** No assembly stall at ADD points
- **Final timestamp:** Expect 15-25% reduction from M11 baseline for medium config

### Benchmark thresholds

Update `tests/python/runtime/test_benchmark.py` thresholds to lock in improvements after each phase.

### Verification checklist

- All numerical outputs identical to pre-M12 (same activations, same weights, same results)
- No message exceeds d_ff elements in payload size
- Color budget (8) not exceeded
- Mesh dimensions smaller than M11 baseline (reduced column heights from RMSNorm simplification)
- Benchmark script reports improved metrics

---

## Out of scope (deferred)

- **Attention direct routing:** Eliminates the 128-element attention collect broadcast but requires 16 colors (exceeds WSE's 8-color limit). Needs a routing approach that doesn't blow the color budget. Potential M13 candidate if a color-efficient routing scheme is found.
- **Congestion-aware routing:** Second-pass route optimization based on link-load estimates.
- **FFN input-dimension tiling:** Tiling FFN-down along d_ff instead of output rows would enable streaming from FFN-up tiles. Changes the linear algebra decomposition.
- **Adaptive XY/YX routing:** Per-route choice between X-first and Y-first.
- **Backpressure / finite buffers:** Per-color input buffer limits and wormhole stall propagation.
- **Operator timing fix:** The operator timing profiling records `end_ts = start_ts + task_base_latency` instead of the actual completion time including element cost. A fix has been prototyped but not yet committed.
