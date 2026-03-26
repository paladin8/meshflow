# Broadcast Messaging

## Goal

Add linear-path broadcast messaging to the mesh runtime, where a single message traverses a path and delivers its payload to every PE along the way — replacing the current approach of N separate point-to-point messages with cloned payloads.

## Motivation

The current runtime broadcasts by emitting N independent messages, each with its own hop list and a full copy of the payload. For a RmsNormReduce broadcasting scale factors to 4 tile PEs in the same column, that's 4 messages traversing overlapping paths. With broadcast messaging, a single message walks the column and delivers at each stop — reducing message count, hop count, and payload cloning.

## Design Decisions

- **Linear paths only** — no tree-shaped multicast. All current broadcast patterns are within a column (same X coordinate).
- **Single payload_slot** — all destinations receive in the same SRAM slot. This matches all current broadcast use cases.
- **`deliver_at` indices on Message** — the hop list stays `Vec<Direction>`. A new `deliver_at: Vec<usize>` field lists hop indices where delivery should occur (see convention below).
- **Automatic detection in the route pass** — the compiler detects when broadcast destinations form a linear column path and generates a single broadcast route instead of N separate routes.
- **Unified route model** — a `BroadcastRoute` dataclass replaces the `(dest_coord, hop_list)` tuples and `payload_slots` lists. Point-to-point is a route with empty `deliver_at`.
- **Scatter unaffected** — scatter sends different data to each destination; broadcast doesn't apply.

---

## `deliver_at` Convention

`deliver_at` lists the hop indices at which the message should deliver its payload **before continuing to forward**. These are intermediate delivery points only. The final delivery (when `is_arrived()` is true) always happens automatically, as today.

- **Empty `deliver_at`**: point-to-point. Deliver only at final destination. Identical to current behavior.
- **Non-empty `deliver_at`**: broadcast. Deliver at each listed hop index (intermediate stops) AND at the final destination (automatic). For N total destination PEs, `deliver_at` has N-1 entries (the intermediate ones); the Nth delivery is the final arrival.

**Index meaning**: `deliver_at` values are **post-arrival** `current_hop` values — the value of `current_hop` when the message arrives at an intermediate PE. When `process_deliver` processes a forwarding step, it advances `current_hop` and forwards the message to the next PE. When that PE receives the message, if `current_hop` matches a `deliver_at` entry, it delivers (writes payload to SRAM, triggers tasks) before continuing to forward.

### Worked Example: Same-column broadcast

Source at (0,0), destinations at (0,1), (0,2), (0,3):

```
Hops: [North, North, North]
deliver_at: [1, 2]   (intermediate deliveries at hop indices 1 and 2)

Step 1: Message at (0,0), current_hop=0.
  - Not arrived (0 < 3). No deliver_at match for 0.
  - Advance: hop[0]=North → forward to (0,1). current_hop becomes 1.

Step 2: Message at (0,1), current_hop=1.
  - Not arrived (1 < 3). current_hop=1 is in deliver_at.
  - DELIVER: write payload to (0,1) SRAM at payload_slot, trigger tasks.
  - Advance: hop[1]=North → forward to (0,2). current_hop becomes 2.

Step 3: Message at (0,2), current_hop=2.
  - Not arrived (2 < 3). current_hop=2 is in deliver_at.
  - DELIVER: write payload to (0,2) SRAM at payload_slot, trigger tasks.
  - Advance: hop[2]=North → forward to (0,3). current_hop becomes 3.

Step 4: Message at (0,3), current_hop=3.
  - Arrived (3 == 3). Final delivery as today.
  - DELIVER: write payload to (0,3) SRAM at payload_slot, trigger tasks.

Result: 3 deliveries from 1 message traversing 3 hops.
```

For N destinations along a path of N hops, `deliver_at` has N-1 entries (indices 1 through N-1), and the Nth destination is the final arrival.

### Convention for inter-column broadcast

Source at (2,0), destinations at (5,0), (5,1), (5,2):

```
Hops: [East, East, East, North, North]   (3 East to reach column 5, then 2 North)
deliver_at: [3, 4]   (deliver after arriving at (5,0) via hop 3, and (5,1) via hop 4)

Step 1-3: Message forwards East through (3,0), (4,0) — no deliveries.
Step 4: Message at (5,0), current_hop=3. deliver_at match → DELIVER + forward North.
Step 5: Message at (5,1), current_hop=4. deliver_at match → DELIVER + forward North.
Step 6: Message at (5,2), current_hop=5. Arrived → final DELIVER.
```

---

## Runtime Changes (Rust)

### Message struct

Add one field to `Message` in `message.rs`:

```rust
pub struct Message {
    // ... existing fields ...
    /// Hop indices for intermediate delivery (see deliver_at convention in spec).
    /// Empty = point-to-point (deliver only at final destination).
    pub deliver_at: Vec<usize>,
}
```

All existing Message construction sites pass `deliver_at: vec![]` to preserve current behavior.

### process_deliver changes

Updated intermediate-hop logic (the forwarding branch):

```
if not arrived:
    if current_hop is in deliver_at:
        write payload.clone() to current PE's SRAM at payload_slot
        trigger waiting tasks on current PE
        increment total_messages
        increment messages_received on current PE
    advance current_hop, compute next PE
    forward message to next PE
```

The `is_arrived` branch (final delivery) is unchanged.

### Profiling

- `total_messages`: incremented once per delivery (intermediate + final). N destinations = N total_messages, same as before.
- `total_hops`: the single message's hop count (path length). Previously N × avg_path_length; now just path_length. **This is the main profiling improvement.**
- `link_counts`: one traversal per link. Previously N overlapping traversals.
- `messages_sent` on source PE: 1 (not N).
- `messages_received` on each destination PE: 1 (same as before).

### emit_broadcast_message helper

New method on `Simulator`:

```rust
fn emit_broadcast_message(
    &mut self,
    timestamp: u64,
    source: Coord,
    dest: Coord,
    hops: Vec<Direction>,
    deliver_at: Vec<usize>,
    payload: Vec<f32>,
    payload_slot: SlotId,
)
```

Creates a Message with `deliver_at` populated and enqueues it. `dest` is the final destination (last PE on the path). Used by the updated `broadcast_to_dests`.

### broadcast_to_dests update

Update signature to accept routes:

```rust
fn broadcast_to_dests(
    &mut self,
    base_time: u64,
    coord: Coord,
    routes: &[BroadcastRouteRuntime],
)
```

Where `BroadcastRouteRuntime` is a new struct:

```rust
struct BroadcastRouteRuntime {
    dest: Coord,
    hops: Vec<Direction>,
    deliver_at: Vec<usize>,
    payload_slot: SlotId,
}
```

For each route, emit a broadcast message (or point-to-point if `deliver_at` is empty). `messages_sent` on the source PE is incremented by `routes.len()` (one message per route, not per destination).

The existing `broadcast_to_dests` call sites in process_execute pass routes from the TaskKind variant fields.

---

## Compiler Changes (Python)

### Schedule IR

New dataclass in `schedule_ir.py`:

```python
@dataclass
class BroadcastRoute:
    hops: list[Direction]
    deliver_at: list[int]    # intermediate hop indices where delivery occurs
    payload_slot: int
```

Update these 5 broadcast-capable task entry types to replace their routing fields with `routes: list[BroadcastRoute]`:

| Entry Type | Old Fields Removed | New Field |
|---|---|---|
| `ConcatCollectForwardEntry` | `route_dests`, `payload_slots` | `routes` |
| `AddEntry` | `output_dests`, `payload_slots` | `routes` |
| `MatMulEntry` | `output_dests`, `payload_slots` | `routes` |
| `RmsNormNormalizeEntry` | `output_dests`, `payload_slots` | `routes` |
| `RmsNormReduceEntry` | `tile_dests`, `scale_slot` | `routes` |

For `RmsNormReduceEntry`: the `scale_slot` field is removed. Each `BroadcastRoute` in `routes` carries `payload_slot=scale_slot`. The route pass populates this when building routes.

All other fields on these entry types (trigger_slot, num_fragments, scatter, etc.) are unchanged.

### Route pass — broadcast detection

New helper in `route.py`:

```python
def _try_linear_broadcast(
    source_coord: tuple[int, int],
    dests: list[tuple[tuple[int, int], list[Direction]]],
    payload_slot: int,
) -> list[BroadcastRoute]:
```

Detection rule:
1. All destinations must share the same X coordinate.
2. **Group destinations by direction from source**: destinations with Y > source_Y go in the "North group", destinations with Y < source_Y go in the "South group", destinations with Y == source_Y are delivered at the column entry point.
3. For each group (North and/or South), sort by distance from source. Compute a single XY route from source to the farthest destination in the group. The path passes through all closer destinations. Compute `deliver_at` indices for each intermediate stop.
4. Return one `BroadcastRoute` per group (1 or 2 routes total).

If condition (1) fails, fall back: return N separate `BroadcastRoute` objects, each with empty `deliver_at` (point-to-point).

If there's only 1 destination, return a single `BroadcastRoute` with empty `deliver_at` (point-to-point).

**Bidirectional case**: If destinations span both sides of the source (some North, some South), this produces 2 broadcast messages — one walking North, one walking South. Each is a valid linear path. This is still better than N separate messages.

### XY routing and column broadcasts

Current XY routing: X hops first (East/West), then Y hops (North/South). For destinations in the same column as the source (same X), the path is pure Y hops. For destinations in a different column, X hops reach the target column first, then Y hops walk the column. The `deliver_at` indices fall within the Y-hop segment.

Destinations that don't share the same X coordinate cannot be served by a single linear broadcast — fall back to N separate point-to-point messages.

### Artifact format

New serde struct in `program.rs`:

```rust
#[derive(Debug, Deserialize)]
struct BroadcastRouteProgram {
    dest: (u32, u32),
    hops: Vec<String>,
    #[serde(default)]
    deliver_at: Vec<usize>,
    payload_slot: u32,
}
```

The 5 task program structs that currently have `route_dests`/`output_dests`/`tile_dests` + `payload_slots`/`scale_slot` fields are updated to `routes: Vec<BroadcastRouteProgram>`. Old artifacts without the new field use `#[serde(default)]` to get empty vecs (point-to-point behavior).

Python `artifact.py` gets a matching `BroadcastRouteTask` dataclass.

### Lower pass

Translates `BroadcastRoute` (schedule IR) → `BroadcastRouteProgram` (artifact). Direction enums to strings. Mechanical.

### Conversion in program.rs

`BroadcastRouteProgram` → `BroadcastRouteRuntime` during artifact loading. Direction strings to `Direction` enum, coord tuple to `Coord`.

---

## Rust TaskKind Changes (pe.rs)

The 5 TaskKind variants that carry routing fields are updated:

| TaskKind Variant | Old Fields Removed | New Field |
|---|---|---|
| `ConcatCollectForward` | `route_dests`, `payload_slots` | `routes: Vec<BroadcastRouteRuntime>` |
| `Add` | `output_dests`, `payload_slots` | `routes: Vec<BroadcastRouteRuntime>` |
| `MatMul` | `output_dests`, `payload_slots` | `routes: Vec<BroadcastRouteRuntime>` |
| `RmsNormNormalize` | `output_dests`, `payload_slots` | `routes: Vec<BroadcastRouteRuntime>` |
| `RmsNormReduce` | `tile_dests`, `scale_slot` | `routes: Vec<BroadcastRouteRuntime>` |

`BroadcastRouteRuntime` is defined in `pe.rs` (or a shared module) since it's part of the task definition.

---

## Files Changed

| File | Change |
|------|--------|
| `message.rs` | Add `deliver_at: Vec<usize>` to Message |
| `pe.rs` | Add `BroadcastRouteRuntime` struct. Update 5 TaskKind variants to use `routes`. |
| `runtime.rs` | Update `process_deliver` for intermediate delivery. Add `emit_broadcast_message`. Update `broadcast_to_dests` signature. Update 5 match arms in `process_execute`. |
| `program.rs` | Add `BroadcastRouteProgram` serde struct. Update 5 task program structs. Add conversion to `BroadcastRouteRuntime`. |
| `schedule_ir.py` | Add `BroadcastRoute` dataclass. Update 5 task entry types. |
| `artifact.py` | Add `BroadcastRouteTask` dataclass. Update 5 task program types. |
| `lower.py` | Translate `BroadcastRoute` → artifact format. Update 5 lowering cases. |
| `route.py` | Add `_try_linear_broadcast`. Update all broadcast-emitting handlers. |

**No changes**: `place.py`, `expand.py`, `graph_ir.py`, `config.py`.

---

## Testing

### Rust unit tests (runtime.rs)

- `test_broadcast_delivery_column`: Message with `deliver_at = [1, 2]` and 3 North hops walks a 3-PE column. All 3 PEs receive payload in SRAM, tasks triggered at each stop.
- `test_broadcast_empty_deliver_at`: Empty `deliver_at` = point-to-point behavior (backward compat).
- `test_broadcast_profiling`: Single broadcast to 3 PEs: `total_messages` = 3, `total_hops` = 3 (not 3+2+1=6), `messages_sent` on source = 1.
- `test_broadcast_payload_not_consumed`: Intermediate deliveries don't modify message payload — last PE gets same data as first.

### Python compiler tests (test_route.py)

- `test_linear_broadcast_detection`: Column destinations → single `BroadcastRoute` with `deliver_at` entries.
- `test_non_column_fallback`: Different-column destinations → N separate `BroadcastRoute` objects with empty `deliver_at`.
- `test_single_dest_no_broadcast`: Single destination → single route with empty `deliver_at`.
- `test_bidirectional_broadcast`: Destinations on both sides of source → 2 `BroadcastRoute` objects (one North, one South).

### Python e2e tests

- All existing transformer block and MLP tests pass with identical numerical outputs (broadcast is transparent).
- `test_broadcast_reduces_hops`: Run a transformer block, compare profiling `total_hops` against pre-broadcast baseline. Verify reduction.

---

## Out of Scope

- **Tree-shaped multicast** — only linear paths. Can be added later for wider mesh layouts.
- **Scatter broadcast** — scatter sends different data per destination; not applicable.
- **Broadcast-aware cost model** — the cost model applies to the producing task, not message transit. No changes needed.
- **Congestion modeling** — broadcast reduces link pressure but we don't model congestion.

---

## Exit Criteria

- `deliver_at` field on Message, with empty = point-to-point behavior.
- `process_deliver` handles intermediate deliveries (write + forward).
- `BroadcastRoute` / `BroadcastRouteRuntime` used throughout schedule IR, artifact, and runtime.
- Route pass automatically detects same-column broadcasts and generates single-message routes, including bidirectional split when destinations span both sides of source.
- All existing tests pass with identical numerical outputs.
- Profiling shows reduced hop counts for broadcast-eligible operators.
- All lints clean.
