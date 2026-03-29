# Milestone 10: Color-Based Routing

## Goal

Add WSE-style color-based routing to model realistic link multiplexing. Colors are integer IDs (0 through K-1) assigned to routes by the compiler and enforced by the runtime. On any given physical link, no two simultaneously-active routes may use the same color. The color budget K defaults to 8 (matching WSE-3).

On WSE hardware, colors serve two purposes:
1. **Link disambiguation** — concurrent flows on the same physical link are separated by color.
2. **Parallel sends** — a PE can emit messages on different colors simultaneously. Only messages on the *same* color are serialized.

The current simulator serializes all sends from a PE regardless (`base_time + i` for the i-th message). This is wrong — it models a PE with a single outgoing channel. With colors, a PE broadcasting to 15 destinations across 8 colors can emit all 8 colors in the first tick and the remaining 7 in the second tick (2 ticks instead of 15). This directly addresses the serialized-send bottleneck identified in M9.

## Sequencing

Five phases, strictly ordered:

0. **Data model** — add `color` field to route structures across Python IR, artifact schema, lowering pass, and Rust runtime. All routes default to color 0. Existing tests pass unchanged.
1. **Conflict graph & greedy coloring** — new `color.py` compiler pass between route and lower. Builds intermediate-PE-overlap conflict graph (required for routing table correctness in Phase 3), assigns colors via greedy coloring, and generates per-PE routing tables. Assert K=8 suffices.
2. **Parallel sends & runtime enforcement** — parallel message emission across colors in `broadcast_to_dests`/`scatter_to_dests`; per-(link, color) `free_at` contention tracking. This is where the performance improvement lands.
3. **Per-PE routing tables** — replace per-message hop lists with per-PE color→direction routing tables. The compiler still computes hop lists internally; the lowering pass converts them into routing table entries on each PE. Messages become lightweight (color + payload, no hop list). This is the correct WSE hardware model.
4. **Profiling & observability** — per-link color sets in `ProfileSummary`. Color utilization metrics in benchmark output.

---

## Phase 0: Data Model

### Naming convention

Two naming patterns for color fields, mirroring the existing `route_dest`/`route_hops` convention:

- **`color`** on `BroadcastRoute`, `BroadcastRouteTask`, `BroadcastRouteRuntime`, `BroadcastRouteProgram`, and `Message` — these are route-level or message-level structures where the context is unambiguous.
- **`route_color`** on single-route task entries/tasks (`ForwardActivationEntry`/`Task`, `LinearEntry`/`Task`, `RmsNormPartialSumEntry`/`Task`) — follows the existing `route_dest`/`route_hops` naming pattern. On the Rust `TaskKind` variants, this is stored as `color` (matching how `route_hops` becomes `hops` in `TaskKind`). The `convert_task` function in `program.rs` maps `route_color` → `color`, as it already maps `route_hops` → `hops`.

### Python Schedule IR changes

**`schedule_ir.py`** — add `color: int = 0` to `BroadcastRoute`:

```python
@dataclass
class BroadcastRoute:
    dest: tuple[int, int] = (0, 0)
    hops: list[Direction] = field(default_factory=list)
    deliver_at: list[int] = field(default_factory=list)
    payload_slot: int = 0
    color: int = 0  # NEW
```

Add `route_color: int = 0` to single-route entries:
- `ForwardActivationEntry`
- `LinearEntry`
- `RmsNormPartialSumEntry` (note: this entry uses `reduce_dest`/`reduce_hops` rather than `route_dest`/`route_hops`, but the color field is still named `route_color` for consistency)

### Python artifact changes

**`artifact.py`** — add `color: int = 0` to `BroadcastRouteTask`. Add `route_color: int = 0` to `ForwardActivationTask`, `LinearTask`, `RmsNormPartialSumTask`.

**Serialization (`_task_to_dict`)**: No changes needed — `asdict(task)` recursively converts all fields, so the new `color` field on `BroadcastRouteTask` and `route_color` on single-route tasks flow through automatically.

**Deserialization (`_dict_to_task`)**: Five `BroadcastRouteTask` reconstruction sites must add `color=r.get("color", 0)`:
1. `concat_collect_forward` branch (line ~286)
2. `add` branch (line ~298)
3. `mat_mul` branch (line ~312)
4. `rms_norm_normalize` branch (line ~328)
5. `rms_norm_reduce` branch (line ~340)

Single-route tasks (`forward_activation`, `linear`, `rms_norm_partial_sum`) use `**fields` construction, so the new `route_color` field flows through automatically with its default value of 0 for pre-M10 artifacts.

### Python lowering changes

**`lower.py`** — `_lower_route` manually enumerates fields and must add `color=route.color`:

```python
def _lower_route(route: BroadcastRoute) -> BroadcastRouteTask:
    return BroadcastRouteTask(
        dest=route.dest,
        hops=[d.value for d in route.hops],
        deliver_at=list(route.deliver_at),
        payload_slot=route.payload_slot,
        color=route.color,  # NEW
    )
```

`_lower_task` manually enumerates fields for each task type. Three branches must add `route_color=task.route_color`:
- `ForwardActivationEntry` branch (line ~54)
- `LinearEntry` branch (line ~67)
- `RmsNormPartialSumEntry` branch (line ~126)

Without these changes, colors assigned by the Phase 1 color pass would be silently dropped during lowering.

### Rust changes

**`message.rs`** — add `pub color: u32` to `Message`:

```rust
pub struct Message {
    // ... existing fields ...
    /// Route color ID for link multiplexing (0 = uncolored).
    pub color: u32,
}
```

Update all `Message` struct literals: `make_message` in tests, `inject_message` in `Simulator`.

**`pe.rs`** — add `pub color: u32` to `BroadcastRouteRuntime`:

```rust
pub struct BroadcastRouteRuntime {
    pub dest: Coord,
    pub hops: Vec<Direction>,
    pub deliver_at: Vec<usize>,
    pub payload_slot: SlotId,
    pub color: u32,  // NEW
}
```

Add `color: u32` to `TaskKind` variants:
- `ForwardActivation` — add `color: u32`
- `Linear` — add `color: u32`
- `RmsNormPartialSum` — add `color: u32`

Update all match arms in `process_execute` that destructure these variants to include the new `color` field and pass it to `emit_message`.

**`program.rs`** — add `#[serde(default)] color: u32` to `BroadcastRouteProgram`. Add `#[serde(default)] route_color: u32` to the `ForwardActivation`, `Linear`, `RmsNormPartialSum` variants of `TaskProgram`.

Update `convert_routes` to carry `color` through to `BroadcastRouteRuntime`. Update `convert_task` to map `route_color` → `color` on `TaskKind` variants (following the existing `route_hops` → `hops` pattern).

**`runtime.rs`** — `emit_message` and `emit_broadcast_message` gain a `color: u32` parameter and set `message.color = color`. Update all call sites:
- `emit_message` called from: `ForwardActivation` arm, `Linear` arm, `RmsNormPartialSum` arm (via `process_execute`), and indirectly via `broadcast_to_dests` and `scatter_to_dests`
- `emit_broadcast_message` called from: `broadcast_to_dests`
- `broadcast_to_dests` and `scatter_to_dests` read `route.color` from `BroadcastRouteRuntime`
- `inject_message` constructs `Message` directly — set `color: 0`
- `add_task` constructs `TaskKind::ForwardActivation` — set `color: 0`

### Config change

**`config.py`** — add `color_budget: int = 8` to `CompilerConfig`.

### Invariant

All routes get color 0. No behavioral change. All existing tests pass unchanged.

### Files changed

| File | Change |
|------|--------|
| `schedule_ir.py` | Add `color` to `BroadcastRoute`, `route_color` to single-route entries |
| `artifact.py` | Add color fields; update 5 `BroadcastRouteTask` reconstruction sites in `_dict_to_task` |
| `lower.py` | Add `color=route.color` to `_lower_route`; add `route_color=task.route_color` to 3 branches in `_lower_task` |
| `config.py` | Add `color_budget: int = 8` |
| `message.rs` | Add `color: u32` to `Message`; update struct literals |
| `pe.rs` | Add `color: u32` to `BroadcastRouteRuntime` and 3 `TaskKind` variants |
| `program.rs` | Add `color` to `BroadcastRouteProgram`, `route_color` to 3 task variants; update converters |
| `runtime.rs` | Thread `color` through `emit_message`, `emit_broadcast_message`, `broadcast_to_dests`, `scatter_to_dests`, `inject_message`, `add_task`; update `process_execute` match arms |

---

## Phase 1: Conflict Graph & Greedy Coloring

### New file: `python/meshflow/compiler/passes/color.py`

The color pass sits between route and lower in the compiler pipeline:

```
expand → place → route → color → lower
```

### Algorithm

1. **Route enumeration**: Walk all `PESchedule` entries in the `ScheduleIR`. For each task, extract every route as a `(source_coord, hops, write_back_ref)` tuple:
   - Multi-route tasks (`ConcatCollectForwardEntry`, `AddEntry`, `MatMulEntry`, `RmsNormNormalizeEntry`, `RmsNormReduceEntry`): extract each `BroadcastRoute` from the `routes` list.
   - Single-route tasks: extract the hop list from `route_hops` (`ForwardActivationEntry`, `LinearEntry`) or `reduce_hops` (`RmsNormPartialSumEntry` — note the different field name).
   - The `write_back_ref` is a `(pe_index, task_index, route_index)` tuple for writing colors back after assignment.

2. **Intermediate PE set extraction**: For each route, compute the set of intermediate PEs it passes through — excluding both the source PE and the final destination. The source PE is excluded because it doesn't consult the routing table; the task itself knows which color to use when emitting the message. The destination PE is excluded because the absence of a routing table entry signals final delivery. Only intermediate forwarding PEs need routing table entries.

   Requires a `_step_coord(coord, direction)` helper since Python coordinates are tuples:
   ```python
   def _step_coord(coord: tuple[int, int], direction: Direction) -> tuple[int, int]:
       x, y = coord
       match direction:
           case Direction.NORTH: return (x, y + 1)
           case Direction.SOUTH: return (x, y - 1)
           case Direction.EAST:  return (x + 1, y)
           case Direction.WEST:  return (x - 1, y)
   ```
   Walk the hop list from the source, collecting PEs after the first hop and before the final destination. The PE set for a route from (0,0) with hops [E, E, N] is {(1,0), (2,0)} — source (0,0) and destination (2,1) are both excluded. A 1-hop route has an empty PE set (no intermediates) and never conflicts with anything.

3. **Conflict graph construction**: Two routes conflict if their intermediate PE sets overlap. This catches routes that pass through the same forwarding PE — which would need the same color→direction routing table entry in Phase 3, but might need different directions. Build via inverted index: for each PE, collect all route indices passing through it; for each pair of routes sharing a PE, add a conflict edge. O(R²) in the worst case but R is small (~100 routes for a transformer block).

   Because the source PE is excluded, multiple routes from the same high-fanout PE (e.g., 17 routes from a broadcast collect PE) do NOT automatically conflict with each other. They only conflict if their paths cross at intermediate PEs. This keeps color pressure manageable — K=8 should suffice unless many routes share the same intermediate corridor.

4. **Greedy coloring**: Sort routes by decreasing conflict degree (most-constrained first). For each route, assign the lowest-numbered color not already used by any conflicting neighbor.

5. **Budget assertion**: If the chromatic number exceeds `config.color_budget`, raise a `CompilerError`. We optimistically assume K=8 suffices for current workloads. If the transformer block budget test fails, the first response is to increase K (a config change); schedule-aware analysis is deferred to a future milestone.

6. **Write back**: Set the `color` field on each `BroadcastRoute` and `route_color` on each single-route task entry, using the `write_back_ref` tuples.

7. **Routing table generation** (used by Phase 3): During step 2, the color pass already walks every route through its intermediate PEs. After color assignment, it builds per-PE routing table entries: for each route, at each intermediate PE, record `(color, direction, optional deliver_slot)`. These entries are stored on `PESchedule.routing_table: dict[int, RouteTableEntry]` — a new field added to the schedule IR. The lower pass mechanically copies this table into the artifact `PEProgram`. This avoids a separate pass re-walking the same routes.

### Compiler pipeline wiring

**`__init__.py`** — import and call `color` pass:

```python
from meshflow.compiler.passes import expand, lower, place, route, color

expanded = expand(graph, config)
spatial = place(expanded, config)
schedule = route(spatial, config, weights)
schedule = color(schedule, config)  # NEW
return lower(schedule, config)
```

### Files changed

| File | Change |
|------|--------|
| `passes/color.py` | New file: route enumeration, PE set extraction, conflict graph, greedy coloring, routing table generation |
| `schedule_ir.py` | Add `routing_table: dict[int, RouteTableEntry]` to `PESchedule`; add `RouteTableEntry` dataclass |
| `__init__.py` | Wire color pass into pipeline |

---

## Phase 2: Parallel Sends & Runtime Enforcement

### Parallel send model

On WSE hardware, a PE can emit one message per color per tick. Messages on different colors depart simultaneously; messages on the same color are serialized. This replaces the current `base_time + i` model.

**`broadcast_to_dests` change** — group routes by color, serialize only within each color:

```rust
fn broadcast_to_dests(
    &mut self,
    base_time: u64,
    coord: Coord,
    routes: &[BroadcastRouteRuntime],
    payload: Vec<f32>,
) {
    self.mesh.pe_mut(coord).counters.messages_sent += routes.len() as u64;
    // Track how many messages per color have been sent so far
    let mut color_counts: HashMap<u32, u64> = HashMap::new();
    for route in routes.iter() {
        let count = color_counts.entry(route.color).or_insert(0);
        let send_time = base_time + *count;  // serialize within same color
        *count += 1;
        // emit message at send_time (same as today per-message)
        ...
    }
}
```

**Example**: 15 routes across 8 colors (2 routes each for colors 0-6, 1 route for color 7):
- Old model: sends at ticks 0, 1, 2, ..., 14 → 15 ticks
- New model: colors 0-6 send at ticks 0 and 1; color 7 sends at tick 0 → 2 ticks

**`scatter_to_dests` change** — same grouping logic. Group by `route.color`, serialize within each color group.

**Single-route tasks** (`ForwardActivation`, `Linear`, `RmsNormPartialSum`): each emits exactly one message, so no parallelism possible. No change needed.

### Backward compatibility with Phase 0

In Phase 0 (before the color pass), all routes have color 0. With the parallel send model, all messages are the same color → all serialized → identical behavior to today. The parallel send model is safe to deploy in Phase 0, but we defer it to Phase 2 so Phases 0 and 1 have zero behavioral change.

### Per-(link, color) contention tracking

Add a `link_color_free_at: HashMap<(Coord, Coord, u32), u64>` to `Simulator`. This tracks when each (directed_link, color) pair becomes free.

### Contention counter

Add `color_contentions: u64` to `ProfileSummary`. This counter is always present (not debug-only), making it testable in both debug and release builds. Tests assert this counter is 0 for correctly-colored programs.

### Forwarding change

In `process_deliver`, when forwarding a message to the next PE, before enqueueing the delivery:

```rust
let key = (coord, neighbor, message.color);
let free_at = self.link_color_free_at.get(&key).copied().unwrap_or(0);
let entry_time = std::cmp::max(timestamp, free_at);

if entry_time > timestamp {
    // Same-color contention detected — compiler produced a bad color assignment.
    self.profile.color_contentions += 1;
}

self.link_color_free_at.insert(key, entry_time + self.config.hop_latency);
self.enqueue_deliver(entry_time + self.config.hop_latency, neighbor, coord, new_message);
```

With correct conservative coloring from Phase 1, `entry_time == timestamp` always — no two same-colored routes share any link. If `color_contentions > 0` after a run, the compiler produced an incorrect assignment.

### Expected impact

The parallel send model directly reduces the serialized-send bottleneck from M9. For the small transformer config, high-fanout PEs like the RmsNorm collect (which broadcasts to Q/K/V tiles) currently serialize all sends. With 8 colors, these sends are parallelized up to 8× per tick. `final_timestamp` should decrease significantly for both configs. Benchmark thresholds will be tightened after measuring the actual improvement.

### Files changed

| File | Change |
|------|--------|
| `runtime.rs` | Parallel send in `broadcast_to_dests` and `scatter_to_dests`; add `link_color_free_at`; contention check in `process_deliver` |
| `profiling.rs` | Add `color_contentions: u64` to `ProfileSummary` |
| `bridge.rs` | Expose `color_contentions` to Python |

---

## Phase 2.1: Color Diversity for Parallel Send Throughput

### Problem

The behavior-based conflict model from Phase 1 minimizes the chromatic number (6 colors for the transformer block) but consolidates routes onto the fewest colors possible. When a collect PE broadcasts to tiles at (5,0), (5,1), (5,2), (5,3), all routes go East first — same direction at each intermediate PE — so they don't conflict and all get color 0. Phase 2's parallel send model then serializes them because they share a color.

### Change

Modify `_greedy_color` in `color.py` to prefer color diversity at each source PE. When assigning a color to a route, avoid colors already used by other routes from the same source PE — not just colors used by conflicting neighbors.

The modified algorithm:

```python
for idx in order:
    forbidden = {colors[n] for n in graph.get(idx, set()) if colors[n] >= 0}
    pe_colors = {colors[j] for j in same_pe_routes[source_pe] if colors[j] >= 0}
    # Prefer a color not forbidden AND not used by same PE
    c = 0
    while c in forbidden or c in pe_colors:
        c += 1
    # If this exceeds budget, fall back to lowest non-forbidden
    if c >= budget:
        c = 0
        while c in forbidden:
            c += 1
    colors[idx] = c
```

This spreads routes from the same PE across distinct colors (up to K), falling back to shared colors only when the budget would be exceeded.

### Example

RmsNorm collect PE broadcasting 3 routes East to Q/K/V columns:
- **Before**: all 3 get color 0 (no conflicts) → 3 serial sends (3 ticks)
- **After**: get colors 0, 1, 2 → 3 parallel sends (1 tick)

### Files changed

| File | Change |
|------|--------|
| `color.py` | Modify `_greedy_color` to accept source PE mapping and color budget; prefer PE-diverse colors |
| `test_color.py` | Add test for color diversity at same-PE routes |
| `test_benchmark.py` | Tighten `final_timestamp` thresholds |

### Expected impact

Significant reduction in `final_timestamp` for both configs — high-fanout PEs now parallelize their sends across K colors instead of serializing on color 0.

---

## Phase 3: Per-PE Routing Tables

### Problem

Messages currently carry their full hop list (`Vec<Direction>`) — the routing lives on the wire. On WSE hardware, routing lives in the mesh: each PE has a routing table mapping `color → direction`. Messages carry only their color; the PE's table determines where to forward them. This phase aligns the simulator with the hardware model.

### Routing table model

Each PE has a routing table: `HashMap<u32, RouteAction>` where:

```rust
enum RouteAction {
    Forward(Direction),
    DeliverAndForward { direction: Direction, deliver_slot: SlotId },
}
```

When a message arrives at a PE:
- **Table has entry for this color** → execute the action (forward, or deliver + forward)
- **No entry** → this is the final destination; deliver payload to `message.payload_slot`

Intermediate broadcast delivery (currently `deliver_at` indices) becomes explicit `DeliverAndForward` entries at intermediate PEs.

### Compiler changes

No new compiler pass changes — the routing tables were already generated by the color pass in Phase 1 (step 7) and stored on `PESchedule.routing_table`. Phase 3 activates their use:

- **`lower.py`**: Mechanically copies `PESchedule.routing_table` into `PEProgram.routing_table`. Strips `hops` and `deliver_at` from `BroadcastRouteTask` (no longer needed — routing is in the tables). Strips `route_hops`/`reduce_hops` from single-route tasks.
- **`color.py`**: No changes (routing tables already generated in Phase 1).

### Scatter route handling

Scatter routes (`ConcatCollectForward` with `scatter=true`) send different payloads to different destinations. Each destination is a separate `BroadcastRoute` with its own color (assigned in Phase 1). The routing table generation in the color pass walks each scatter route independently, creating per-PE entries for each color. Messages from `scatter_to_dests` carry the per-destination color. No special handling beyond normal per-route table generation is needed.

### Artifact format changes

**`artifact.py`** — `PEProgram` gains `routing_table: dict[int, RouteTableEntry]`:

```python
@dataclass
class RouteTableEntry:
    direction: str  # "north", "south", "east", "west"
    deliver_slot: int | None = None  # if set, deliver to this slot before forwarding
```

`BroadcastRouteTask` simplifies — loses `hops` and `deliver_at`, keeps `dest`, `payload_slot`, `color`:

```python
@dataclass
class BroadcastRouteTask:
    dest: tuple[int, int] = (0, 0)
    payload_slot: int = 0
    color: int = 0
```

Single-route tasks (`ForwardActivationTask`, `LinearTask`, `RmsNormPartialSumTask`) lose their `route_hops`/`reduce_hops` fields. They keep `route_dest`/`reduce_dest` (for profiling) and `route_color`.

### Rust runtime changes

**`message.rs`** — `Message` sheds `hops`, `current_hop`, `deliver_at`. Adds `hop_count: u32` (incremented on each forward, used for `total_hops` profiling):

```rust
pub struct Message {
    pub id: u64,
    pub source: Coord,
    pub dest: Coord,
    pub payload: Vec<f32>,
    pub payload_slot: SlotId,
    pub timestamp: u64,
    pub color: u32,
    pub hop_count: u32,  // incremented each forward, for profiling
}
```

**`pe.rs`** — `PE` gains `routing_table: HashMap<u32, RouteAction>`. `BroadcastRouteRuntime` simplifies (loses `hops`, `deliver_at`). `TaskKind` variants lose their `hops`/`route_hops`/`reduce_hops` fields.

**`program.rs`** — `PEProgram` serde struct gains `routing_table`. `BroadcastRouteProgram` loses `hops`, `deliver_at`. Task variants lose hop fields. `convert_task` and `convert_routes` simplified accordingly.

**`runtime.rs`** — `process_deliver` changes from hop-list routing to table lookup:

```rust
fn process_deliver(&mut self, timestamp: u64, coord: Coord, message: &mut Message) {
    let pe = self.mesh.pe_mut(coord);
    pe.counters.messages_received += 1;

    if let Some(action) = pe.routing_table.get(&message.color).cloned() {
        // Routing table path
        match action {
            RouteAction::Forward(dir) => {
                let neighbor = coord.step(dir, ...);
                message.hop_count += 1;
                // link tracking, contention check, enqueue delivery
            }
            RouteAction::DeliverAndForward { direction, deliver_slot } => {
                pe.write_slot(deliver_slot, message.payload.clone());
                self.profile.total_messages += 1;
                // trigger tasks at this PE
                let neighbor = coord.step(direction, ...);
                message.hop_count += 1;
                // link tracking, contention check, enqueue delivery
            }
        }
    } else {
        // No routing entry → final destination
        let payload = std::mem::take(&mut message.payload);
        pe.write_slot(message.payload_slot, payload);
        self.profile.total_messages += 1;
        self.profile.total_hops += message.hop_count as u64;
        // trigger tasks
    }
}
```

**`emit_message` and `emit_broadcast_message`** — no longer set hops on the message. Just set color, payload, payload_slot, dest.

### Test/injection backward compatibility

The Rust test helpers (`inject_message`, `add_task` in `runtime.rs`, and `bridge.rs`) don't use compiled routing tables. Changes:

- **`inject_message`**: Creates self-delivery messages (source == dest, 0 hops). These work unchanged — no routing table entry means immediate final delivery. Just remove `hops`/`current_hop`/`deliver_at` from the `Message` literal.
- **`add_task`** (`InjectTaskKind::ForwardActivation`): Currently computes hops via `generate_route_xy` and stores them in `TaskKind::ForwardActivation { hops, ... }`. Must be rewritten to compute routing table entries on intermediate PEs instead. The `InjectTaskKind::ForwardActivation` variant loses its implicit hop computation; the caller provides a color, and `add_task` populates routing table entries along the path.
- **`bridge.rs`**: Uses `inject_message` for input injection (self-delivery, no change needed). If it uses `add_task`, the same changes apply.

### Validation

The lowering pass validates that no PE has conflicting routing table entries (two routes requesting different directions for the same color at the same PE). With the intermediate-PE-overlap conflict graph from Phase 1, this should never happen — but the validation catches bugs.

### Files changed

| File | Change |
|------|--------|
| `lower.py` | Copy `routing_table` from `PESchedule` to `PEProgram`; strip hop fields from routes/tasks |
| `artifact.py` | Add `RouteTableEntry`, `routing_table` on `PEProgram`; simplify `BroadcastRouteTask`; remove hop fields from single-route tasks |
| `message.rs` | Remove `hops`, `current_hop`, `deliver_at`; add `hop_count` |
| `pe.rs` | Add `RouteAction`, `routing_table` on `PE`; simplify `BroadcastRouteRuntime` and `TaskKind` variants |
| `program.rs` | Add routing table serde; simplify route/task serde structs; update converters |
| `runtime.rs` | `process_deliver` uses table lookup; update `emit_message`/`emit_broadcast_message`; rewrite `add_task` for routing table setup |
| `bridge.rs` | Update if it uses `add_task`; `inject_message` path unchanged |

---

## Phase 4: Profiling & Observability

### Profiling additions

**`profiling.rs`** — add to `ProfileSummary`:

```rust
/// Per-link set of distinct colors that traversed it during simulation.
pub link_color_sets: HashMap<(Coord, Coord), HashSet<u32>>,
/// Maximum number of distinct colors on any single link.
pub max_colors_per_link: u32,
/// Total distinct colors used across all routes.
pub total_colors_used: u32,
```

Track in `process_deliver`: when a message traverses a link, insert its color into `link_color_sets`. Compute `max_colors_per_link` and `total_colors_used` at simulation end (in `run()`, before returning `SimResult`).

### Bridge exposure

Expose `max_colors_per_link`, `total_colors_used`, and `color_contentions` through the Python bridge.

### Benchmark script updates

**`scripts/benchmark.py`** — add to the metrics table:
- `total_colors_used`: number of distinct colors assigned by the compiler
- `max_colors_per_link`: maximum color count on any single link
- `color_contentions`: runtime contention count (should be 0)

### Benchmark regression test updates

**`tests/python/runtime/test_benchmark.py`** — add assertions:
- `total_colors_used <= config.color_budget` (colors fit within budget)
- `color_contentions == 0` (no same-color contention at runtime)

### Files changed

| File | Change |
|------|--------|
| `profiling.rs` | Add `link_color_sets: HashSet`, `max_colors_per_link`, `total_colors_used` |
| `runtime.rs` | Record color in `link_color_sets` during `process_deliver`; compute summary at end |
| `bridge.rs` | Expose `max_colors_per_link`, `total_colors_used` to Python |
| `scripts/benchmark.py` | Report color metrics |
| `tests/python/runtime/test_benchmark.py` | Assert color budget compliance and zero contentions |

---

## Files Changed (Summary)

| File | Phase | Change |
|------|-------|--------|
| `schedule_ir.py` | 0, 1 | Add `color` to `BroadcastRoute`, `route_color` to entries; add `RouteTableEntry`, `routing_table` on `PESchedule` |
| `artifact.py` | 0, 3 | Add color fields + 5 deser sites; then add `RouteTableEntry`, `routing_table`, simplify route/task structs |
| `lower.py` | 0, 3 | Add color to `_lower_route` + 3 `_lower_task` branches; then copy routing table, strip hop fields |
| `config.py` | 0 | Add `color_budget: int = 8` |
| `message.rs` | 0, 3 | Add `color: u32`; then remove `hops`/`current_hop`/`deliver_at`, add `hop_count` |
| `pe.rs` | 0, 3 | Add `color` to route/task structs; then add `RouteAction`, `routing_table`, simplify structs |
| `program.rs` | 0, 3 | Add color to serde structs; then add routing table serde, simplify route/task structs |
| `runtime.rs` | 0, 2, 3, 4 | Thread color; parallel sends + `link_color_free_at`; table-based `process_deliver` + `add_task` rewrite; profiling |
| `bridge.rs` | 2, 3, 4 | Expose `color_contentions`; update if uses `add_task`; expose color profiling fields |
| `passes/color.py` | 1 | New: intermediate-PE-overlap conflict graph + greedy coloring |
| `__init__.py` | 1 | Wire color pass into pipeline |
| `profiling.rs` | 2, 4 | Add `color_contentions`; add `link_color_sets`, `max_colors_per_link`, `total_colors_used` |
| `scripts/benchmark.py` | 4 | Report color metrics |
| `tests/python/runtime/test_benchmark.py` | 4 | Assert color budget compliance and zero contentions |

**No changes**: `expand.py`, `place.py`, `route.py`, `graph_ir.py`, `spatial_ir.py`.

---

## Testing

### Phase 0 tests

- All existing tests pass unchanged (color defaults to 0 everywhere).
- New unit test: `BroadcastRoute` and `BroadcastRouteTask` round-trip with non-zero color.
- New unit test: `ForwardActivationTask`, `LinearTask`, `RmsNormPartialSumTask` round-trip with non-zero `route_color`.
- New Rust test: `Message` creation with color field, `BroadcastRouteRuntime` with color.

### Phase 1 tests (`tests/python/compiler/passes/test_color.py`)

- `test_no_conflict_all_color_zero`: routes on non-overlapping PEs all get color 0.
- `test_shared_pe_different_colors`: two routes sharing an intermediate PE get different colors.
- `test_greedy_coloring_order`: most-constrained route is colored first.
- `test_budget_exceeded_raises`: artificially create more conflicts than K → `CompilerError`.
- `test_transformer_block_within_budget`: full transformer block compiles with ≤8 colors (small + medium configs).
- `test_color_assignment_idempotent`: running color pass twice produces the same assignment.

### Phase 2 tests

- `test_parallel_send_different_colors`: 3 routes on 3 different colors depart at the same tick (send_time difference = 0).
- `test_serial_send_same_color`: 3 routes on the same color depart at ticks 0, 1, 2 (serialized).
- `test_mixed_parallel_serial`: routes on 2 colors (4 on color 0, 2 on color 1) → color 0 takes 4 ticks, color 1 takes 2 ticks, total = 4 ticks (not 6).
- `test_no_contention_with_correct_coloring`: full transformer block runs, `color_contentions == 0`.
- `test_contention_detected`: manually inject two same-colored messages on the same link at the same time → `color_contentions > 0`.
- `test_final_timestamp_decreases`: transformer block `final_timestamp` is lower than M9 baseline (parallel sends reduce critical path).
- Rust unit test: `link_color_free_at` tracking with sequential and concurrent messages.

### Phase 3 tests

- `test_routing_table_generated`: compiled transformer block has non-empty routing tables on intermediate PEs.
- `test_routing_table_no_conflicts`: no PE has two different directions for the same color.
- `test_message_no_hops`: messages in table-based path have empty hops / no hop list.
- `test_hop_count_matches`: `message.hop_count` at delivery matches expected Manhattan distance.
- `test_routing_table_correctness`: full transformer block end-to-end numerical correctness with routing tables.
- `test_broadcast_intermediate_delivery`: broadcast route with `deliver_at` produces `DeliverAndForward` entries at correct PEs.
- Rust unit test: `RouteAction` lookup, `process_deliver` with routing table.

### Phase 4 tests

- `test_color_profiling_fields`: `max_colors_per_link`, `total_colors_used` are populated after a run.
- `test_benchmark_color_metrics`: benchmark script outputs color metrics.
- Benchmark regression: both configs compile and run within K=8 budget.

### Numerical correctness

All existing tests (95 Rust, 300+ Python) must pass unchanged after each phase. Phases 0 and 1 are pure metadata additions with no behavioral change. Phase 2 changes message timing (parallel sends) but not computation. Phase 3 changes how routing is performed (table lookup vs hop list) but not the routes themselves. Outputs must match the Python reference at the same numerical tolerance throughout.

---

## Out of Scope

- **Schedule-aware conflict analysis** — using task ordering to reduce false conflicts. Only needed if K=8 proves too tight. First response to budget exhaustion is increasing K (config change), not implementing schedule analysis.
- **Cross-color bandwidth sharing** — N colors on a link each getting 1/N throughput. Future fidelity improvement.
- **Automatic overflow handling** — re-routing or time-phasing when colors exceed budget. We assert and fail instead.
- **Link utilization heatmap visualization** — can be added as a follow-on using the profiling data from Phase 4.
- **Color-aware route optimization** — re-routing to reduce color pressure. Future milestone.

---

## Exit Criteria

- Color field propagates end-to-end: Python ScheduleIR → lower → artifact → Rust Message.
- Greedy coloring assigns distinct colors to routes with overlapping PE sets.
- Transformer block (small + medium configs) compiles within K=8 color budget.
- Parallel sends across colors: messages on different colors depart in the same tick; same-color messages serialize.
- `final_timestamp` decreases for both configs (parallel sends reduce critical path).
- Runtime `color_contentions` counter is 0 for correctly-colored programs.
- Profiling exposes per-link color sets and max colors per link.
- Benchmark script reports color utilization metrics.
- Benchmark metrics improve monotonically across phases for both configs.
- Per-PE routing tables replace per-message hop lists; messages carry only color + payload.
- All existing tests pass unchanged (numerical correctness preserved).
- All lints clean.
