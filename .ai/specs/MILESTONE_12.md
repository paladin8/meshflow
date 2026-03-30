# Milestone 12: Routing & Placement Optimizations

## Goal

Reduce link contention and peak per-message payload sizes through two compiler optimizations that target the bottlenecks exposed by M11's wormhole routing model.

**Motivation:** M11 made payload size a first-class cost driver. Profiling reveals two structural problems:

1. **Hot corridor:** All collect PEs sit at the same row (the column midpoint), so every inter-layer broadcast flows through the same east-west corridor. The hottest links carry 8–12 messages in the small/medium configs.

2. **Attention output payload blowup:** The attention collect PE concatenates all seq_len position outputs into a single `seq_len × d_model` payload before broadcasting to the projection layer. This produces the largest message in the system — 32 elements (small) / 128 elements (medium) — blocking a single physical link for that many cycles. This grows as O(seq_len × d_model).

```
Current bottleneck (small config, row 2):

  (1,2)──6 msgs──>(2,2)──8 msgs──>(3,2)──8 msgs──>(4,2)──8 msgs──>(5,2)
   CC               CC               CC               CC               CC
   │                │                │                │                │
   fan N/S          fan N/S          fan N/S          fan N/S          fan N/S
```

## Key design decisions

1. **Three-way staggered collect rows (Phase 1):** Cycle the collect PE's Y-position across three positions — `center - 1`, `center`, `center + 1` — based on `col % 3`. This distributes east-west broadcast traffic across three rows instead of one. Two-row stagger is insufficient: with 10 active columns, 5 per row still creates contention, and adjacent columns whose broadcasts overlap (e.g., RMSNorm col 1 traversing cols 2–4, and K-linear col 3 traversing col 4–5) can land on the same row. Three-way ensures that no two adjacent columns share a stagger position. The cost is minimal — collect moves at most ±1 from center, adding at most 1 extra hop for the furthest tile. This is a placement-only change; XY routing and broadcast optimization adapt automatically since they operate on concrete coordinates.

2. **Attention direct routing (Phase 2):** Eliminate the attention collect PE. Route each attention PE's AV output (d_model elements) directly to the downstream projection tiles. Each projection tile gathers seq_len fragments via a new `ConcatCollectLocal` task, then triggers the existing Linear computation. This replaces one 32/128-element broadcast with seq_len independent d_model-element broadcasts that naturally travel on different rows (since attention PEs are vertically distributed).

3. **New `ConcatCollectLocal` runtime task:** A minimal variant of ConcatCollect that writes the concatenated result to a specified SRAM output_slot (triggering downstream tasks) instead of writing to the simulation output map. Reuses the existing `process_concat_fragment` accumulator logic. This is the only runtime change in the milestone.

4. **Slot allocation for projection tiles with collect:** Currently, projection tiles use slots 0 (input), 1 (weight), 2 (bias). With direct routing, slots 0..seq_len-1 receive attention PE fragments, the ConcatCollectLocal accumulates internally, then writes the concatenated result to a designated output slot that triggers the Linear task. Weight and bias slots shift to seq_len and seq_len+1.

## Sequencing

Two phases, strictly ordered:

1. **Staggered collect row placement** — Modify `_middle_collect_rows` to accept a stagger offset. Update `_place_tiled_compute_group`, `_place_rmsnorm_group`, and `_place_attention_group` to pass column index as stagger offset. Re-baseline contention thresholds. Verify numerical correctness unchanged.

2. **Attention direct routing** — Remove attention collect PE from placement. Generate direct routes from attention PEs to projection tiles. Add `ConcatCollectLocal` task to runtime. Add ConcatCollectLocal tasks on projection tiles. Re-baseline payload sizes and contention thresholds. Verify numerical correctness unchanged.

---

## Phase 1: Staggered Collect Row Placement

### Current behavior

`_middle_collect_rows(num_tiles)` in `place.py:301-315` always places the collect PE at `num_tiles // 2`:

```python
def _middle_collect_rows(num_tiles: int) -> tuple[list[int], int]:
    collect_row = num_tiles // 2
    tile_rows = []
    for i in range(num_tiles):
        tile_rows.append(i if i < collect_row else i + 1)
    return tile_rows, collect_row
```

For a 4-tile column (height 5), collect is always at row 2:
```
row 0: tile 0
row 1: tile 1
row 2: COLLECT  ← always here
row 3: tile 2
row 4: tile 3
```

Every column in the mesh gets the same layout, creating the row-2 hot corridor.

### New behavior

Add a `stagger_offset` parameter with 3-way cycling:

```python
def _middle_collect_rows(num_tiles: int, stagger_offset: int = 0) -> tuple[list[int], int]:
    base_collect = num_tiles // 2
    # 3-way stagger: offset cycles through -1, 0, +1
    delta = (stagger_offset % 3) - 1  # yields -1, 0, 1
    collect_row = base_collect + delta
    # Clamp to valid range [0, num_tiles]
    collect_row = max(0, min(collect_row, num_tiles))
    tile_rows = []
    for i in range(num_tiles):
        tile_rows.append(i if i < collect_row else i + 1)
    return tile_rows, collect_row
```

With `col % 3`, collect cycles through three Y positions:
- `col % 3 == 0`: collect at `center - 1` (one row north of center)
- `col % 3 == 1`: collect at `center` (baseline position)
- `col % 3 == 2`: collect at `center + 1` (one row south of center)

For a 4-tile column (center = 2):
```
col%3=0:              col%3=1:              col%3=2:
row 0: tile 0         row 0: tile 0         row 0: tile 0
row 1: COLLECT        row 1: tile 1         row 1: tile 1
row 2: tile 1         row 2: COLLECT        row 2: tile 2
row 3: tile 2         row 3: tile 2         row 3: COLLECT
row 4: tile 3         row 4: tile 3         row 4: tile 3
```

Small config column assignments (`col % 3`):
```
col 0(FW):0  col 1(RN):1  col 2(Q):2  col 3(K):0  col 4(V):1
col 5(At):2  col 6(Pr):0  col 7(RN):1  col 8(FF↑):2  col 9(FF↓):0
```
No two adjacent columns share the same stagger position.

### Callers to update

**`_place_tiled_compute_group`** (place.py:318-370): Pass `col` as `stagger_offset`:
```python
tile_rows, collect_row = _middle_collect_rows(len(group.tiles), stagger_offset=col)
```

**`_place_rmsnorm_group`** (place.py:373-474): Same change. Note: the reduce PE stays at `num_tiles + 1` (above all tiles and collect), independent of stagger. Only the collect PE and tile rows shift.
```python
tile_rows, collect_row = _middle_collect_rows(group.num_tiles, stagger_offset=col)
```

**`_place_attention_group`** (place.py:477-537): **Excluded from stagger.** Attention groups use an N-to-1 gather pattern (seq_len attention PEs → collect), which is sensitive to collect distance. Moving the collect away from center increases the number of PEs on the longer side, adding link serialization delay on the gather path. The attention collect stays at center (`stagger_offset=1` → delta=0):
```python
pe_rows, collect_row = _middle_collect_rows(group.seq_len, stagger_offset=1)
```

### Impact on broadcast optimization

The broadcast optimization in `_try_linear_broadcast` and `_broadcast_single_column` (route.py:576-697) groups destinations by `dest_x` and computes XY hops from the actual source coordinate. It does not assume any particular Y position for the source. Since it operates on concrete coordinates from the spatial IR, staggering requires no changes to the routing pass.

### Expected traffic distribution

Small config (before → after):
```
Before: All east traffic on row 2
  (2,2)->(3,2): 8 msgs   (3,2)->(4,2): 8 msgs

After: East traffic split across rows 1, 2, 3
  col%3=0 broadcasts on row 1, col%3=1 on row 2, col%3=2 on row 3
  Peak messages per link: ~3 (⅓ of original)
```

### Testing

- **Numerical correctness:** All transformer block outputs must match reference implementation (unchanged). The stagger changes routing distances slightly but computation is identical.
- **Contention reduction:** `link_contentions` should decrease. Re-baseline the benchmark thresholds downward.
- **No height increase:** Column heights remain `num_tiles + 1` (or `num_tiles + 2` for RmsNorm). Mesh dimensions should not grow.
- **Broadcast optimization still works:** Verify routing table entries are correct and column broadcasts deliver to the right intermediate PEs.

---

## Phase 2: Attention Direct Routing

### Current behavior

The attention output flows through a central collect PE:

```
Attention PEs (one per seq position)
    │ d_model elements each
    ▼
Attention Collect PE
    │ concatenates seq_len fragments
    │ broadcasts seq_len × d_model elements
    ▼
Projection Linear Tiles
```

For the small config (seq_len=4, d_model=8):
- 4 attention PEs each send 8 elements to collect at (5,2)
- Collect concatenates into 32 elements
- Broadcasts 32-element payload to projection tiles at column 6 via (5,2)->(6,2)
- Single link occupancy: 32 cycles per broadcast × 2 broadcasts (N/S) = 64 link-cycles on (5,2)->(6,2)

For the medium config (seq_len=8, d_model=16):
- Concatenated payload: 128 elements
- Single link occupancy: 128 cycles per broadcast

### New behavior

Each attention PE routes directly to the projection tiles:

```
Attention PEs (one per seq position)
    │ d_model elements each
    │ routed directly (XY from each PE's row)
    ▼
Projection Linear Tiles (each with ConcatCollectLocal)
    │ gathers seq_len × d_model elements
    │ triggers Linear computation
    ▼
Projection Collect PE
```

Traffic now uses different physical links since attention PEs are at different rows:
```
Attn PE at (5,0) → east on row 0 → column 6 tiles
Attn PE at (5,1) → east on row 1 → column 6 tiles
Attn PE at (5,3) → east on row 3 → column 6 tiles
Attn PE at (5,4) → east on row 4 → column 6 tiles
```

Per-message payload: d_model (8 or 16) instead of seq_len × d_model (32 or 128).
Peak link occupancy: d_model cycles instead of seq_len × d_model.

### Placement changes

**`_place_attention_group`** (place.py:477-537):

When `has_collect=True` (seq_len > 1 and av_matmul_id is not None), currently creates a collect PE and internal edges from attention PEs to collect.

Change: **Do not create the attention collect PE or internal attention→collect edges.** The attention PEs remain at their current positions. The edges from attention PEs to downstream nodes (projection layer) become inter-group edges handled by `_generate_inter_group_edges`.

The attention group's `node_pe_map` no longer includes the collect PE. The group height decreases by 1 (or stays the same if the collect PE was at the same height as attention PEs).

**Edge routing impact:** The expanded IR's `AttentionGroup` currently has edges: `attn_pe_i → collect_id` (internal) and `collect_id → projection_tiles` (inter-group). With direct routing:
- Remove internal `attn_pe_i → collect_id` edges
- Replace the single inter-group `collect_id → projection_tiles` edge with `seq_len` inter-group edges: `attn_pe_i → projection_group` for each i

This requires changes to the expanded IR's `AttentionGroup` to support a "no-collect" mode, or changes to `_generate_inter_group_edges` to handle the attention-to-projection edge pattern specially.

### Routing changes

**`_route_attention_pe`** (route.py:437-525):

Currently, AV MatMul output routes follow outgoing inter-group edges. With the collect PE removed, outgoing edges go directly to projection tiles.

The AV MatMul task's routes change from:
- 1 route to collect PE (payload_slot = position_index)

To:
- N routes to projection tiles, one per tile in the projection column
  - Use broadcast optimization: 2 routes (N/S broadcast in projection column)
  - payload_slot encodes the position index (for the projection tile's ConcatCollect)

**Projection tile pre-task:**

Each projection tile needs a ConcatCollect to gather seq_len fragments before the Linear task fires.

Slot allocation per projection tile:
```
Slots 0..seq_len-1: attention PE fragments (one per position)
Slot seq_len:       weight matrix (W)
Slot seq_len+1:     bias vector (b)
Slot seq_len+2:     (reserved for ConcatCollect output → Linear input)
```

Task chain on each projection tile:
1. `ConcatCollectLocal` tasks (trigger_slot=0..seq_len-1, num_fragments=seq_len, output_slot=seq_len+2)
   - When all fragments arrive, writes concatenated result to output_slot
   - This triggers the downstream Linear task
2. `Linear` task (trigger_slot=seq_len+2, input_slot=seq_len+2, weight_slot=seq_len, bias_slot=seq_len+1)
   - Fires when ConcatCollectLocal writes the concatenated input

### Runtime changes

Add `ConcatCollectLocal` variant to `TaskKind`:
```rust
TaskKind::ConcatCollectLocal {
    num_fragments: u32,
    total_rows: u32,
    fragment_offset: u32,
    fragment_rows: u32,
    num_positions: u32,
    output_slot: SlotId,
}
```

Execution logic reuses `process_concat_fragment`. When the final fragment arrives and `Some(result)` is returned, call `task_write_slot(timestamp, coord, output_slot, result)` to write the concatenated data to SRAM and trigger downstream tasks. This is ~10 lines of new runtime code.

Corresponding changes:
- **`pe.rs`**: Add `ConcatCollectLocal` to the `TaskKind` enum.
- **`program.rs`**: Add deserialization for `"concat_collect_local"` kind in artifact loading.
- **`bridge.rs`**: No changes needed (ConcatCollectLocal is internal, not exposed to Python sim API).
- **`artifact.py`**: Add `ConcatCollectLocalTask` dataclass and serialization/deserialization support.
- **`_mesh_runtime.pyi`**: Add `ConcatCollectLocal` to `TaskKind` enum if exposed.

### Color assignment impact

With direct routing, each attention PE broadcasts to the projection column. With seq_len=4, that's 4 × 2 = 8 broadcast routes (2 per PE for N/S broadcast). These routes share intermediate PEs in the projection column, so the color pass may need more colors.

However, since each broadcast arrives from a different direction (different source rows), the routing table entries at projection column PEs will have different directions, making them naturally conflict (different colors required). The existing greedy coloring handles this; the color budget of 8 should be sufficient for seq_len ≤ 8.

### Handling edge cases

- **seq_len == 1:** Only one attention PE, no collect PE needed (already the case). Direct routing is trivially a single route. No change needed.
- **No AV MatMul:** If the attention group has no av_matmul_id (QK-only attention), there's no output to route. No change needed.
- **Attention group feeds non-linear node:** If the downstream node is not a LinearGroup (unlikely in current architecture), fall back to the current collect-based approach.

### Expected improvement

Small config:
- Payload per broadcast: 32 → 8 elements (4× reduction)
- Peak link occupancy: 32 cycles → 8 cycles per broadcast
- Traffic on (5,2)→(6,2): eliminated (no collect PE at row 2)
- Traffic distributed across rows 0,1,3,4

Medium config:
- Payload per broadcast: 128 → 16 elements (8× reduction)
- Peak link occupancy: 128 cycles → 16 cycles per broadcast
- Dramatic reduction in link wait cycles for attention→projection stage

### Testing

- **Numerical correctness:** Transformer block outputs must match reference (identical computation, different message flow).
- **Payload sizes:** No message in the system should exceed d_ff elements (the FFN intermediate, now the largest payload).
- **Contention:** Link contentions should decrease further from Phase 1 baseline.
- **Slot allocation:** Verify projection tiles have correct slot assignments (weights, biases shifted to accommodate fragment slots).
- **Color budget:** Verify total_colors_used stays within budget of 8.

---

## Re-baseline (after both phases)

### Updated profiling expectations

After both optimizations:
- **Hot corridor eliminated:** No single row carries all inter-layer traffic
- **Attention payload capped at d_model:** Largest payloads are now FFN broadcasts (d_ff elements)
- **Link contentions significantly reduced:** Expect 40-60% reduction from current M11 baseline
- **Final timestamp:** May decrease modestly (5-15%) due to reduced contention wait cycles

### Benchmark thresholds

Update `tests/python/runtime/test_benchmark.py` thresholds to lock in improvements. Tighten:
- `link_contentions` for both small and medium configs
- `final_timestamp` if measurably improved

Update `tests/python/runtime/test_color_routing.py`:
- `test_transformer_block_final_timestamp_reasonable` threshold
- `test_small_transformer_link_contentions` threshold
- `test_medium_transformer_link_contentions` threshold

### Verification checklist

- All numerical outputs identical to pre-M12 (same activations, same weights, same results)
- Link contention reduced from M11 baseline
- No message exceeds d_ff elements in payload size
- Color budget (8) not exceeded
- Mesh dimensions no larger than M11 baseline
- Benchmark script reports improved metrics

---

## Out of scope (deferred)

- **Congestion-aware routing:** Second-pass route optimization based on link-load estimates. Requires a link traffic model and iterative route refinement. Natural M13 candidate.
- **FFN payload optimization:** The FFN-up → FFN-down broadcast (d_ff elements) is now the largest payload. Tiling along the d_ff dimension could reduce this but changes the linear algebra decomposition.
- **Adaptive XY/YX routing:** Per-route choice between X-first and Y-first based on estimated congestion. Conflicts with current broadcast chain structure.
- **Backpressure / finite buffers:** Per-color input buffer limits and wormhole stall propagation.
