# Milestone 1: Minimal Mesh Simulator

## Objective

Build the Rust runtime skeleton: an event-driven mesh simulator that can route messages across a 2D grid of processing elements, execute a trivial task, and report profiling counters. Includes a minimal PyO3 bridge so Python tests can drive the simulator.

Note: The overall design's M1 exit criteria mention only Rust unit tests. This spec deliberately pulls the PyO3 bridge into M1 so that Python integration tests can exercise the runtime early, consistent with CLAUDE.md's rule to "run at least one Python end-to-end test against the runtime" when changing runtime logic.

## Exit criteria

1. A message can enter a source PE, traverse a multi-hop route, and arrive at a destination PE.
2. A `ForwardActivation` task can receive a message, copy its payload, and emit it onward along a route.
3. A forwarding chain of 3+ PEs delivers a payload end-to-end with correct data.
4. Route latency, hop counts, and per-PE message stats are reported.
5. Rust unit tests cover all core types and the event loop.
6. Python integration tests exercise the simulator through the PyO3 bridge.
7. `cargo test -p mesh_runtime` and `uv run pytest tests/python` both pass.

---

## 1. Rust module layout

```
crates/mesh_runtime/src/
├── lib.rs          # PyO3 module registration, re-exports
├── coords.rs       # Coord type, Direction enum, stepping logic
├── message.rs      # Message struct, MessageId
├── pe.rs           # PE struct: SRAM blob store, input queue, task configs
├── mesh.rs         # Mesh struct: 2D grid of PEs, lookup by Coord
├── route.rs        # HopList type, dimension-ordered (XY) route generator
├── task.rs         # TaskKind enum, TaskConfig, task execution logic
├── event.rs        # Event struct, EventKind enum, EventQueue (min-heap)
├── runtime.rs      # Simulator: config, event loop, result collection
├── profiling.rs    # PeCounters, ProfileSummary
└── bridge.rs       # PyO3-exposed classes and functions
```

---

## 2. Core data structures

### 2.1 Coordinates and directions

```rust
struct Coord {
    x: u32,
    y: u32,
}

enum Direction {
    North,  // y + 1
    South,  // y - 1
    East,   // x + 1
    West,   // x - 1
}
```

`Coord::step(direction) -> Option<Coord>` returns `None` if the step would go out of bounds (requires knowing mesh dimensions, or handled at the mesh level).

### 2.2 Message

```rust
struct Message {
    id: u64,
    source: Coord,
    dest: Coord,
    hops: Vec<Direction>,
    current_hop: usize,
    payload: Vec<f32>,
    payload_slot: SlotId,
    timestamp: u64,
}
```

- `payload` carries the tensor data inline with the message as it traverses the mesh. This is the simplest approach for M1 — data travels with the message, not via cross-PE SRAM references.
- `hops` is a pre-computed hop list from the route generator.
- `current_hop` tracks progress through the hop list.
- `payload_slot` identifies which SRAM slot the payload is written to upon final delivery at the destination PE.
- `source` and `dest` are retained for debugging and profiling. They are not consumed by the event loop — routing is driven entirely by the hop list.
- When `current_hop == hops.len()`, the message has arrived at its destination.

### 2.3 SlotId

```rust
type SlotId = u32;
```

Convention for M1:
- Slot 0 = task input
- Slot 1 = task output

### 2.4 Processing element

```rust
struct PE {
    coord: Coord,
    sram: HashMap<SlotId, Vec<f32>>,
    input_queue: VecDeque<Message>,
    tasks: Vec<TaskConfig>,
    sram_capacity_bytes: Option<usize>,  // not enforced in M1
    counters: PeCounters,
}
```

SRAM operations:
- `write_slot(slot_id, data: Vec<f32>)` — insert or overwrite
- `read_slot(slot_id) -> &Vec<f32>` — panic if missing (missing = compiler bug)
- `remove_slot(slot_id)` — free a slot after consumption (optional in M1)

### 2.5 Mesh

```rust
struct Mesh {
    width: u32,
    height: u32,
    pes: Vec<PE>,  // flat storage, indexed by y * width + x
}
```

Constructed with `Mesh::new(width, height)`. All access goes through accessor methods — never index `pes` directly:
- `mesh.pe(coord) -> &PE`
- `mesh.pe_mut(coord) -> &mut PE`
- Internal index: `coord.y * self.width + coord.x`

### 2.6 Task

```rust
enum TaskKind {
    ForwardActivation {
        input_slot: SlotId,
        route_dest: Coord,
        hops: Vec<Direction>,  // pre-computed at setup time, not during execution
    },
    CollectOutput {
        input_slot: SlotId,
    },
}

struct TaskConfig {
    kind: TaskKind,
    trigger_slot: SlotId,  // task fires when this slot is written
}
```

- `ForwardActivation`: reads `input_slot` and emits a message to `route_dest` along the pre-computed `hops` list. The hop list is generated once at task configuration time (see section 4.1.1), not during event processing. No local output slot is written in M1 — the data goes directly into the outbound message.
- `CollectOutput`: reads `input_slot`, stores it as a simulation output. No outbound message.

A task fires when its `trigger_slot` is written by an incoming message.

### 2.7 Event system

```rust
struct Event {
    timestamp: u64,
    kind: EventKind,
    // tie-breaking fields for determinism
    coord: Coord,
    sequence: u64,  // global monotonic counter for insertion order
}

enum EventKind {
    DeliverMessage { message: Message },
    ExecuteTask { task_index: usize },
}
```

`EventQueue` wraps a `BinaryHeap<Reverse<Event>>` (min-heap). Ordering: `(timestamp, coord.y, coord.x, sequence)` for fully deterministic execution.

### 2.8 Profiling

```rust
struct PeCounters {
    messages_received: u64,
    messages_sent: u64,
    tasks_executed: u64,
    slots_written: u64,
}

struct ProfileSummary {
    total_messages: u64,
    total_hops: u64,
    total_events_processed: u64,
    total_tasks_executed: u64,
    final_timestamp: u64,
    per_pe: HashMap<Coord, PeCounters>,
}
```

### 2.9 Simulation result

```rust
struct SimResult {
    outputs: HashMap<Coord, Vec<f32>>,
    profile: ProfileSummary,
}
```

### 2.10 Simulator config

```rust
struct SimConfig {
    width: u32,
    height: u32,
    hop_latency: u64,        // default: 1
    task_base_latency: u64,  // default: 1
    max_events: u64,         // default: 100_000, safety limit
}
```

---

## 3. Routing

### 3.1 Route generator

A standalone function:

```rust
fn generate_route_xy(from: Coord, to: Coord) -> Vec<Direction>
```

Algorithm: move along X first (East if `to.x > from.x`, West otherwise), then Y (North if `to.y > from.y`, South otherwise).

Example: `(0,0)` to `(3,2)` produces `[East, East, East, North, North]`.

Special case: `from == to` returns an empty hop list.

### 3.2 Route execution

The runtime consumes hop lists, not coordinates. It does not call the generator during event processing:

```
next_direction = message.hops[message.current_hop]
neighbor = mesh.step(current_coord, next_direction)
message.current_hop += 1
enqueue DeliverMessage at neighbor
```

### 3.3 Pluggability

The route generator is a plain function. Swapping to a different algorithm later means replacing the function call where hop lists are built. No trait abstraction needed in M1.

### 3.4 Edge cases

- Same-PE message: empty hop list, payload is written directly to the local PE's SRAM slot and tasks are triggered immediately.
- Out-of-bounds hop: panic. This indicates a bug in route generation, not a runtime condition.

---

## 4. Event loop

### 4.1 Simulator lifecycle

1. **Configure** — Create `Mesh` from `SimConfig`. Configure tasks on PEs.
2. **Inject** — Seed input messages (see 4.1.1).
3. **Run** — Process events until the queue is empty or `max_events` is reached.
4. **Collect** — Return `SimResult` with outputs and profiling.

#### 4.1.1 Injection semantics

When a message is injected (via `SimInput.add_message` from Python, or programmatically in Rust tests), the following happens:

1. Generate a hop list: `hops = generate_route_xy(source, dest)`.
2. Build a `Message` with the payload inline and the generated hop list.
3. If `hops` is empty (same-PE delivery): enqueue `DeliverMessage` at the source coord at timestamp 0.
4. Otherwise: enqueue `DeliverMessage` at the source coord at timestamp 0. The event loop will handle forwarding from there.

**Task hop lists** are pre-computed at task configuration time (via `SimInput.add_task` or programmatically in Rust tests), not during event processing. When a `ForwardActivation` task is configured on a PE, `generate_route_xy(pe_coord, route_dest)` is called immediately and the resulting hop list is stored in the `TaskKind::ForwardActivation` variant. This keeps the event loop free of route generation calls, consistent with section 3.2.

### 4.2 Event processing loop

The event loop distinguishes between **intermediate hops** (forwarding through a PE) and **final delivery** (message has reached its destination). Only final delivery writes to SRAM and triggers tasks. Intermediate PEs never see the payload.

```
global_sequence = 0

while let Some(event) = queue.pop():
    if profile.total_events_processed >= config.max_events:
        break  // safety limit

    match event.kind:
        DeliverMessage { message }:
            pe = mesh.pe_mut(event.coord)
            pe.counters.messages_received += 1

            if message.current_hop < message.hops.len():
                // INTERMEDIATE HOP: forward to next PE without touching SRAM
                next_dir = message.hops[message.current_hop]
                message.current_hop += 1
                neighbor = mesh.step(event.coord, next_dir)
                pe.counters.messages_sent += 1
                queue.push(Event {
                    timestamp: event.timestamp + config.hop_latency,
                    kind: DeliverMessage { message },
                    coord: neighbor,
                    sequence: global_sequence++,
                })
            else:
                // FINAL DELIVERY: write payload to SRAM, trigger tasks
                pe.write_slot(message.payload_slot, message.payload)
                pe.counters.slots_written += 1
                profile.total_messages += 1
                profile.total_hops += message.hops.len()

                for (i, task) in pe.tasks.iter().enumerate():
                    if task.trigger_slot == message.payload_slot:
                        queue.push(Event {
                            timestamp: event.timestamp + config.task_base_latency,
                            kind: ExecuteTask { task_index: i },
                            coord: event.coord,
                            sequence: global_sequence++,
                        })

        ExecuteTask { task_index }:
            pe = mesh.pe_mut(event.coord)
            task = &pe.tasks[task_index]
            match task.kind:
                ForwardActivation { input_slot, route_dest, hops }:
                    data = pe.read_slot(input_slot).clone()
                    // build outbound message with pre-computed hop list
                    new_message = Message {
                        payload: data,
                        hops: hops.clone(),
                        current_hop: 0,
                        payload_slot: 0,  // convention: deliver to slot 0
                        ...
                    }
                    if hops.is_empty():
                        // same-PE: deliver locally
                        enqueue DeliverMessage at event.coord at current timestamp
                    else:
                        enqueue DeliverMessage at event.coord at current timestamp
                        // first DeliverMessage will see current_hop=0 < hops.len()
                        // and forward to the first neighbor
                    pe.counters.messages_sent += 1
                    pe.counters.tasks_executed += 1
                    profile.total_tasks_executed += 1

                CollectOutput { input_slot }:
                    data = pe.read_slot(input_slot).clone()
                    outputs.insert(event.coord, data)
                    pe.counters.tasks_executed += 1
                    profile.total_tasks_executed += 1

    profile.total_events_processed += 1
    profile.final_timestamp = event.timestamp
```

### 4.3 Timing model

All costs are unitless ticks for relative comparison:

| Parameter | Default | Meaning |
|---|---|---|
| `hop_latency` | 1 | Cost per mesh hop |
| `task_base_latency` | 1 | Cost to execute any task |
| `max_events` | 100,000 | Safety limit on total events processed |

### 4.4 Determinism

Events with equal timestamps are ordered by `(timestamp, coord.y, coord.x, sequence)`. The `sequence` field is a global monotonic counter incremented on every enqueue, ensuring stable ordering even for events at the same PE and timestamp.

---

## 5. PyO3 bridge

### 5.1 Python-facing API

**`MeshConfig` class:**

```python
cfg = MeshConfig(width=4, height=4)
cfg.hop_latency = 1
cfg.task_base_latency = 1
cfg.max_events = 100_000
```

**`SimInput` class:**

```python
inp = SimInput()
inp.add_message(source=(0, 0), dest=(3, 2), payload=[1.0, 2.0, 3.0])
inp.add_task(coord=(0, 0), kind=TaskKind.ForwardActivation,
             trigger_slot=0, route_dest=(3, 2))
inp.add_task(coord=(3, 2), kind=TaskKind.CollectOutput, trigger_slot=0)
```

**`run_simulation` function:**

```python
result = run_simulation(config=cfg, inputs=inp)
result.outputs         # dict: {(x, y): list[float]}
result.total_hops      # int
result.total_messages  # int
result.final_timestamp # int
result.pe_stats        # dict: {(x, y): PeStats}
```

**`PeStats` class:**

```python
stats = result.pe_stats[(0, 0)]
stats.messages_received  # int
stats.messages_sent      # int
stats.tasks_executed     # int
```

### 5.2 Design principles

- **One call does the whole simulation.** Python builds config + inputs, calls `run_simulation`, gets results. No per-step callbacks across the boundary.
- **Route generation stays in Rust.** `add_message` takes source/dest coordinates. Rust generates the hop list internally. Python never sees hop lists.
- **Tuples for coordinates.** Python passes `(x, y)` tuples, Rust converts to `Coord`.
- **Enum for task kinds.** Python uses `TaskKind.ForwardActivation` or `TaskKind.CollectOutput` — a PyO3-exposed IntEnum. Invalid types are caught by PyO3 at the call boundary.
- **Errors become Python exceptions.** Invalid inputs (out-of-bounds coordinates, unrecognized task kind strings, empty payloads) raise `ValueError` via PyO3's error conversion. Simulation invariant violations (e.g., missing SRAM slot) raise `RuntimeError`.

### 5.3 What is not exposed in M1

- No streaming or incremental simulation
- No direct PE inspection from Python
- No mesh mutation after creation
- No compiled artifact loading (M2/M3)

---

## 6. Test plan

### 6.1 Rust unit tests

**`coords` tests:**
- Direction stepping produces correct neighbor coordinates
- Stepping out of mesh bounds returns `None`

**`route` tests:**
- XY routing: `(0,0)` to `(3,2)` → `[E, E, E, N, N]`
- Same-PE: `(1,1)` to `(1,1)` → empty list
- Adjacent PEs: `(0,0)` to `(1,0)` → `[E]`
- Reverse direction: `(3,2)` to `(0,0)` → `[W, W, W, S, S]`

**`pe` tests:**
- Slot write/read round-trip
- Read missing slot panics
- Slot overwrite replaces data
- Input queue ordering (FIFO)
- Task trigger matching: task fires when matching slot is written

**`event` tests:**
- Min-heap ordering by timestamp
- Tie-breaking: same timestamp orders by `(y, x, sequence)`

**`mesh` tests:**
- Construction with given dimensions, all PEs initialized
- PE lookup by valid coordinate
- PE lookup by invalid coordinate panics or errors

### 6.2 Rust integration tests (in `runtime` module)

**Single hop:**
- Message from `(0,0)` to `(1,0)`, verify delivery, hop count = 1, final timestamp = hop_latency

**Multi-hop:**
- Message from `(0,0)` to `(3,2)`, verify delivery, hop count = 5, final timestamp = 5 * hop_latency + task_base_latency (for CollectOutput)

**ForwardActivation chain:**
- `(0,0)` → ForwardActivation → `(2,0)` → ForwardActivation → `(4,0)` → CollectOutput
- Verify payload `[1.0, 2.0, 3.0]` arrives intact at `(4,0)`

**Multiple concurrent messages:**
- Two messages seeded at different source PEs targeting different sinks
- Both arrive with correct payloads
- Deterministic event ordering verified

**Profiling accuracy:**
- Run a known scenario, verify all counters (total_hops, total_messages, per-PE stats) match expected values

**Termination safety:**
- Verify the simulator respects `max_events` and terminates even if the queue is not empty

### 6.3 Python integration tests

**`tests/python/test_mesh_bridge.py`:**

**Smoke test:**
- Create `MeshConfig(width=2, height=2)`, run simulation with no messages, verify zero counters

**Single message delivery:**
- Send payload `[1.0, 2.0, 3.0]` from `(0,0)` to `(2,2)` on a 4x4 mesh
- Verify `result.outputs[(2,2)] == [1.0, 2.0, 3.0]`
- Verify `result.total_hops == 4`

**Forward chain:**
- Set up `(0,0)` → forward → `(2,0)` → forward → `(4,0)` → collect on a 5x1 mesh
- Verify output payload matches input

**Profiling from Python:**
- Run a scenario, verify hop counts, message counts, and per-PE stats are accessible and correct

---

## 7. Deliverables summary

| Deliverable | Location | Description |
|---|---|---|
| Coord, Direction types | `coords.rs` | 2D coordinate and direction primitives |
| Message type | `message.rs` | Activation payload carrier with hop list |
| PE with SRAM | `pe.rs` | Processing element with blob store and task configs |
| Mesh grid | `mesh.rs` | 2D grid of PEs |
| XY route generator | `route.rs` | Dimension-ordered hop list generator |
| Task system | `task.rs` | ForwardActivation and CollectOutput |
| Event queue | `event.rs` | Min-heap with deterministic tie-breaking |
| Simulator | `runtime.rs` | Event loop, config, result collection |
| Profiling | `profiling.rs` | Per-PE and global counters |
| PyO3 bridge | `bridge.rs` | MeshConfig, SimInput, run_simulation, SimResult |
| Rust tests | `crates/mesh_runtime/src/*.rs` | Unit tests per module + integration tests |
| Python tests | `tests/python/test_mesh_bridge.py` | Integration tests through PyO3 |

## 8. What is explicitly not in M1

- No compiled artifact format or loading
- No weight pre-loading
- No real compute tasks (no matrix ops)
- No compiler integration
- No API server changes
- No trace event export (only summary counters)
- No SRAM capacity enforcement
- No adaptive or alternative routing algorithms
