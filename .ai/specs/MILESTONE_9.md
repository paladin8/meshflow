# Milestone 9: Compiler Performance Optimizations

## Goal

Reduce message count, hop count, and end-to-end latency through compiler placement and routing optimizations. No new operators or model capabilities — pure performance for existing graphs.

## Baseline (pre-M9)

Canonical transformer block benchmark (seq_len=4, d_model=8, d_ff=16, mesh_height=6):

| Metric | Baseline |
|--------|----------|
| Mesh dimensions | 13 × 6 = 78 PEs |
| Active PEs | 57 (73%) |
| Total messages | 114 |
| Total hops | 331 |
| Avg hops/message | 2.9 |
| Broadcast routes | 31 |
| Point-to-point routes | 118 |
| Max hops per route | 8 |
| Final timestamp | 737 |
| Total tasks executed | 77 |
| Max sends (single PE) | 30 at (1,5) |
| Max queue depth | 15 at (1,5) |
| Hottest link | (1,5)→(2,5): 15 msgs |

## Sequencing

Five phases, strictly ordered:

0. **Benchmark infrastructure** — canonical benchmark script + regression test + log directory.
1. **Multi-column broadcast grouping** — enhance broadcast detection to group by destination column.
2. **Compact placement** — merge single-PE groups into adjacent columns to reduce mesh width.
3. **Middle collect PE placement** — place collect PEs at the center of their tiles to reduce internal hops and distribute horizontal traffic.
4. **RmsNorm+Linear co-location** — co-locate RmsNorm normalize and downstream Linear on the same PE, eliminating the RmsNorm collect PE entirely.

---

## Phase 0: Benchmark Infrastructure

### Canonical configs

| Config | seq_len | d_model | d_ff | mesh_height | Purpose |
|--------|---------|---------|------|-------------|---------|
| `small` | 4 | 8 | 16 | 6 | Current standard, fast iteration |
| `medium` | 8 | 16 | 32 | 8 | Tests generalization to larger dimensions |

Both use `weights_seed=42`, `input_seed=99`.

### Benchmark script

**File**: `scripts/benchmark.py`

Runs both configs through compile → serialize → run_program. Prints a comparison table of all metrics. Accepts `--label` to tag the run.

Appends results to `artifacts/benchmarks/benchmark_log.jsonl` — one JSON object per run with:
- `timestamp` (ISO 8601)
- `label` (from `--label` flag)
- `git_commit` (current HEAD short SHA)
- Per-config metrics: mesh dimensions, active PEs, total_messages, total_hops, avg_hops_per_message, broadcast_routes, point_to_point_routes, max_hops, final_timestamp, total_tasks_executed, max_sends, max_sends_pe, max_queue_depth, max_queue_depth_pe, hottest_link, hottest_link_count

### Regression test

**File**: `tests/python/runtime/test_benchmark.py`

Asserts key metrics don't exceed the baseline for the small config. Thresholds are tightened after each phase to lock in improvements.

### Output directory

`artifacts/benchmarks/` — created by the benchmark script. Contains `benchmark_log.jsonl`.

---

## Phase 1: Multi-Column Broadcast Grouping

### Problem

`_try_linear_broadcast` requires all destinations to share the same X coordinate. When a ConcatCollectForward PE broadcasts to tiles across multiple columns (e.g., RmsNorm collect at x=1 broadcasting to Q/K/V tiles at x=2,3,4), it falls back to N point-to-point messages. PE (1,5) sends 15 individual messages because of this.

### Change

Enhance `_try_linear_broadcast` in `route.py` to:

1. Group destinations by `(dest_x, payload_slot)` — destinations in different columns or with different payload slots cannot share a broadcast message.
2. Within each group, apply the existing column-broadcast logic (sort by distance, build `deliver_at` indices for intermediates, farthest destination is the final arrival).
3. Return the combined result from all groups.

The grouping key includes `payload_slot` because destinations in different columns often receive into different slots (e.g., Q tiles at slot 0, K tiles at slot 1, V tiles at slot 2). Within a column, all tiles share the same payload_slot, so the per-group broadcast works as today.

### Example

RmsNorm collect at (1,5) broadcasting to Q/K/V tiles:

**Before**: 15 point-to-point messages (5 tiles × 3 columns).

**After**: 3 broadcast messages (one per target column), each delivering to ~5 tiles along the way.

### Files changed

`route.py` only — ~20 lines modifying `_try_linear_broadcast`.

### Expected impact

Reduces sends from high-fanout PEs by ~5× for typical transformer blocks.

---

## Phase 2: Compact Placement

### Problem

The current placer assigns one column per group, incrementing X monotonically. The transformer block uses 13 columns, but some groups are single-PE (FORWARD, ADD, COLLECT_SIMPLE) that waste an entire column. This creates long horizontal routes — max 8 hops, average 4.6 hops per route.

### Change

After the initial column-per-group layout, run a compaction pass that merges single-PE groups into adjacent columns where there's vertical space available.

Greedy column merging in `place.py`:

1. Lay out groups as today (column-per-group).
2. Identify single-PE groups (FORWARD, ADD, COLLECT_SIMPLE) that occupy a column alone.
3. For each single-PE group, check if the preceding column has an unused row. If so, place the PE there and eliminate the empty column.
4. Reassign X coordinates to close gaps.

### Constraints

- Only merge single-PE groups — multi-PE groups (tiled linear, RmsNorm, attention) need their own column for vertical stacking.
- Merged PEs must not conflict with existing PEs at the same coordinate.
- A single-PE group at column C can only merge into column C-1 (the immediately preceding column), provided there is a free row. It must not merge forward into column C+1, as that would place it ahead of its consumers in the mesh layout.
- The first column (C=0) is never a merge candidate — it has no preceding column to merge into.

### Invariant preservation

`_find_first_layer_origin` in `route.py` assumes the first LINEAR tile is at x=0. This invariant is preserved because: (1) multi-PE groups (like LINEAR) are never merge candidates, so they keep their own columns; (2) the FORWARD group at x=0 has no preceding column to merge into, so it stays at x=0; (3) after gap-closing, the first LINEAR group shifts left but retains its relative position. No changes to `route.py` are needed in this phase.

### Files changed

`place.py` — new `_compact_columns` pass applied after `_place_columns`.

### Expected impact

Reduce mesh width from 13 to ~9–10 columns, reducing average and max hop distances.

---

## Phase 3: Middle Collect PE Placement

### Problem

All tile PEs in a column start at row 0 and stack upward, with the collect PE on top. This creates two issues:

1. **Internal gather latency**: the farthest tile is N hops from the collect PE.
2. **Top-row congestion**: collect PE broadcasts exit along the top row, concentrating link traffic on one horizontal path.

### Change

Place collect PEs at the center row of their tiles. Tiles split above and below the collect PE.

**Layout for 4 tiles + collect:**

```
Current:          Middle collect:
row 4: collect    row 4: tile_3
row 3: tile_3     row 3: tile_2
row 2: tile_2     row 2: collect    ← max 2 hops from any tile
row 1: tile_1     row 1: tile_1
row 0: tile_0     row 0: tile_0
```

### Benefits

- **Internal gather latency halved**: max tile→collect distance drops from N to ⌈N/2⌉.
- **Traffic distribution**: outgoing broadcasts leave from the center of the mesh instead of the top edge, naturally spreading across rows.
- **No separate staggering needed**: different columns with different tile counts automatically have collect PEs at different Y positions.
- **Compatible with Phase 2 compaction**: single-PE groups merged into a column in Phase 2 occupy unused rows. Phase 3 changes which rows tiles and collect PEs occupy, but the total column height is unchanged, so previously-merged single-PE groups retain available rows.

### Files changed

`place.py` — modify `_place_tiled_compute_group`, `_place_rmsnorm_group`, and `_place_attention_group` to split tiles around a center row for the collect PE.

### Expected impact

Reduced internal hop counts and more even link utilization across rows.

---

## Phase 4: LinearCollect+Add Co-location

### Problem

In the transformer block, each residual add receives two inputs: the residual path (from FORWARD or a previous ADD) and the output of a Linear collect PE (from `out_proj` or `ffn2`). Currently these are separate PEs — the collect PE concatenates tile fragments, then sends a message to the ADD PE. This message adds one hop + delivery latency per residual connection.

### Pattern

Softmax and MatMul are already co-located on attention PEs. Softmax writes to a local SRAM slot, which triggers the AV MatMul task via `task_write_slot`. No message needed between them.

### Change

Place the Linear collect PE and its downstream ADD PE on the same coordinate. The ConcatCollectForward task's route to the ADD becomes a 0-hop self-delivery — the message is created but delivered instantly on the same PE with no hop latency. The residual input still arrives via message as today. This requires no runtime changes — the existing message delivery mechanism handles 0-hop routes naturally.

### Applicability

Two instances in the transformer block:
- `out_proj` collect + `add1` (attention residual)
- `ffn2` collect + `add2` (FFN residual)

The optimization applies when a ConcatCollectForward PE has exactly one downstream ADD PE as its sole non-broadcast destination. The collect's broadcast routes to other downstream operators (if any) are unchanged.

### SRAM slot conflict resolution

The collect PE uses slots 0..N-1 for fragment gathering. The ADD normally uses slots 0 and 1 for its two inputs. To avoid conflicts, edges targeting co-located ADD nodes have their `dst_slot` offset by N (the collect's `num_fragments`). The ADD sees its inputs at slots N and N+1, with output at slot N+2. The route pass's `_route_add` reads these remapped slots naturally.

### Placement

The ADD PE is eliminated as a separate node. Instead, the collect PE absorbs the add functionality. In `_place_columns`, when we encounter a PassthroughGroup(ADD) whose sole input is a preceding collect PE, we skip placing it and mark the collect PE as hosting the add.

This interacts with Phase 2 compact placement: the ADD was previously a single-PE group that sometimes merged into a neighbor. Now it doesn't exist as a separate node — it lives on the collect PE.

### Files changed

- `place.py` — detect collect→add pairs, skip placing separate ADD PE, co-locate on collect PE coordinate.
- `route.py` — generate Add tasks on the collect PE, triggered by a local SRAM slot write instead of a message. Route residual input to the collect PE's add slot.
- `expand.py` — update node expansions so ADD node maps to the collect PE coordinate.

### Expected impact

Eliminates 2 messages (one per residual connection) and 2 columns (ADD PEs merged into collect PEs). Reduces final_timestamp by removing 2 hops from the critical path.

---

## Files Changed (Summary)

| File | Phase | Change |
|------|-------|--------|
| `scripts/benchmark.py` | 0 | New benchmark script |
| `tests/python/runtime/test_benchmark.py` | 0 | New regression test |
| `route.py` | 1 | Enhance `_try_linear_broadcast` to group by `(dest_x, payload_slot)` |
| `place.py` | 2 | Add `_compact_columns` pass |
| `place.py` | 3 | Middle collect PE placement in `_place_tiled_compute_group`, `_place_rmsnorm_group`, `_place_attention_group` |
| `expand.py` | 4 | Update node expansions for collect+add co-location |
| `place.py` | 4 | Detect collect→add pairs, skip separate ADD PE |
| `route.py` | 4 | Generate Add tasks on collect PE with local SRAM triggers |

**No changes**: `runtime.rs`, `message.rs`, `pe.rs`, `program.rs`, `bridge.rs`, `artifact.py`, `lower.py`, `graph_ir.py`, `config.py`.

---

## Testing

### Benchmark tests (`test_benchmark.py`)

- Regression assertions on both configs (small + medium), tightened after each phase.
- Benchmark log entry per phase in `artifacts/benchmarks/benchmark_log.jsonl`.

### Phase 1 tests (`test_route.py`)

- `test_multi_column_broadcast_grouping`: destinations across 3 columns → 3 broadcast routes (not 15 point-to-point).
- `test_multi_column_mixed_slots`: different payload_slots in the same column produce separate groups.

### Phase 2 tests

- `test_compact_single_pe_merged`: single-PE groups (ADD, FORWARD) merge into adjacent columns.
- `test_compact_preserves_multi_pe_columns`: tiled groups keep their own columns.
- `test_compact_no_coordinate_conflicts`: no two PEs at the same coordinate.

### Phase 3 tests

- `test_middle_collect_placement`: collect PE is at the center row of its tiles.
- `test_middle_collect_reduces_max_internal_hops`: max tile→collect distance ≤ ⌈N/2⌉.

### Phase 4 tests

- `test_collect_add_colocation`: co-located PE has both ConcatCollectForward and Add tasks.
- `test_add_pe_eliminated`: no separate ADD PE in the spatial IR when preceded by a collect PE.
- `test_colocation_numerical_correctness`: transformer block outputs match reference with co-location enabled.

### Numerical correctness

All existing tests (95 Rust, 278+ Python) must pass unchanged after each phase. The optimizations are transparent — same computation, different placement and routing.

---

## Out of Scope

- **RmsNorm+Linear co-location** — RmsNorm tiles produce feature fragments, but Linear tiles need the full input vector. Co-location requires a streaming dot product (accumulating partial products as fragments arrive), which needs a new Rust task kind. Deferred to a future milestone.
- **ConcatCollect elimination** — N×M messages (fragment-to-fragment) is worse than N+2 (gather + broadcast). Not worth the complexity.
- **Parallel sends from a single PE** — runtime change, not a compiler optimization.
- **Tree-shaped multicast** — only linear-path broadcast. Can be added later.
- **Congestion-aware routing** — broadcast reduces link pressure but we don't model congestion.
- **New operators or model capabilities** — this milestone is purely about performance.

---

## Exit Criteria

- Benchmark infrastructure captures metrics for both configs across all phases.
- Multi-column broadcast grouping reduces sends from high-fanout PEs.
- Compact placement reduces mesh width from 13 to ~9–10 columns.
- Middle collect PE placement reduces max internal tile→collect hops by ~50%.
- LinearCollect+Add co-location eliminates separate ADD PEs and their inbound messages.
- Benchmark metrics improve monotonically across phases for both configs.
- `final_timestamp` decreases for both configs.
- All existing tests pass unchanged (numerical correctness preserved).
- All lints clean.
