# Milestone 11: Fabric Bandwidth (Wormhole Routing)

## Goal

Model realistic WSE fabric bandwidth by making payload size a first-class cost driver. Currently, every hop costs a fixed 1 tick regardless of payload size — a 4-element activation and a 4096-element activation traverse the mesh identically. On real WSE hardware, the fabric carries 1 word (16 bits) per cycle per physical link. A payload of N elements occupies a link for N cycles.

This milestone replaces the fixed `hop_latency` with a wormhole routing model:

- **Link occupancy = payload size**: A message of N elements occupies each link for N cycles. This is the core fidelity improvement — placement quality, tile sizing, and route length all become directly observable in timing.
- **Wormhole pipelining**: The head flit traverses each hop in 1 cycle. Subsequent flits follow in pipeline. A message traversing H hops arrives after `N + H - 1` cycles (not `N * H`). Links along the path are occupied for overlapping, staggered windows.
- **Per-link contention (color-independent)**: Colors multiplex on the same physical link. All messages sharing a physical link queue behind each other regardless of color. Colors remain important for routing table disambiguation but do not provide independent bandwidth.
- **Data width**: 1 f32 element = 1 cycle (modeling the fabric as carrying one logical element per cycle, matching WSE's fp16 native width).

## Key design decisions

1. **Wormhole routing, not store-and-forward**: Messages pipeline across hops. The head flit proceeds to the next link after 1 cycle; the tail flit follows N-1 cycles later. This matches WSE hardware.

2. **Per-link bandwidth (colors share)**: The physical link carries 1 word/cycle total. Multiple colors on the same link interleave — they do not get independent bandwidth. `link_free_at` is keyed by `(source, dest)` link, not `(source, dest, color)`.

3. **Hop-by-hop event model**: Keep the existing Deliver-event-per-hop architecture. At each intermediate PE, the head flit triggers a Deliver event after 1 cycle. The PE forwards immediately (wormhole — no buffering). Link occupancy is set to `payload.len()` cycles. This approach keeps contention detection physically grounded — links are reserved when messages actually reach them, not predicted in advance.

4. **Tail-flit completion delay**: At the final destination (and for DeliverAndForward intermediate copies), the full payload is not available until the tail flit arrives: `payload.len() - 1` cycles after the head flit. Task triggering and SRAM writes wait for this.

5. **No backpressure (deferred)**: Input queues remain unbounded. Wormhole stall propagation (blocked head flit freezing upstream links) is not modeled. This is a natural follow-on milestone.

6. **Natural broadcast serialization**: Remove explicit per-color send serialization from `broadcast_to_dests` / `scatter_to_dests`. All sends emitted at `base_time`. Link contention tracking naturally serializes messages sharing the same outgoing link.

## Sequencing

Five phases, strictly ordered:

0. **Data model** — Remove `hop_latency` from SimConfig, bridge, and artifact. Change `link_color_free_at` to `link_free_at` (per-link, not per-link-color). Rename `color_contentions` → `link_contentions`, add `total_link_wait_cycles`. Fix compilation, update tests.
1. **Wormhole timing in `forward_message`** — Link occupancy = `entry_time + payload.len()`. Head flit arrival = `entry_time + 1`. Contention against per-link `free_at`.
2. **Tail-flit delivery delay** — Final destination tasks trigger at `arrival + payload.len() - 1 + task_base_latency`. DeliverAndForward SRAM write at `arrival + payload.len() - 1`, head flit forwarding at `arrival`. Empty payloads (len=0) have no tail-flit delay.
3. **Remove broadcast/scatter serialization** — Remove `color_counts` from `broadcast_to_dests` and `scatter_to_dests`. All sends at `base_time`. Link contention handles serialization.
4. **Re-baseline benchmarks and profiling** — Update timing-dependent test expectations. Re-baseline benchmark thresholds. Report `link_contentions` and `total_link_wait_cycles`. Verify numerical correctness unchanged.

---

## Phase 0: Data Model

### Rust changes

**`runtime.rs` — `SimConfig`**: Remove `hop_latency: u64` field and its default. Head flit latency is always 1 cycle (hardcoded constant).

**`runtime.rs` — `Simulator`**: Change field:
```rust
// Before (M10):
link_color_free_at: HashMap<(Coord, Coord, u32), u64>,

// After (M11):
link_free_at: HashMap<(Coord, Coord), u64>,
```

**`profiling.rs` — `ProfileSummary`**: Rename and add:
```rust
// Before:
pub color_contentions: u64,

// After:
pub link_contentions: u64,
pub total_link_wait_cycles: u64,
```

Keep `link_color_sets` and `max_colors_per_link` and `total_colors_used` — they remain useful for observability.

**`bridge.rs`**:
- `MeshConfig` pyclass: Remove `hop_latency` field and its `__new__` parameter.
- `SimResult` pyclass: Rename `color_contentions` → `link_contentions`. Add `total_link_wait_cycles: u64`.
- `sim_result_to_py` function: Update to populate the renamed/new fields from `ProfileSummary`.

**`program.rs`**: In `MeshConfigProgram` (the serde struct), keep `hop_latency` as `#[serde(default)]` but do not pass it to `SimConfig` (which no longer has the field). This provides backward compatibility for pre-M11 artifacts.

### Python changes

**`artifact.py` — `MeshProgramConfig`**: Remove `hop_latency` field. Update serialization/deserialization to omit it. For backward compatibility with pre-M11 artifacts, `_dict_to_config` should silently ignore a `hop_latency` key if present.

**`_mesh_runtime.pyi`**: Remove `hop_latency` from `MeshConfig` class and its `__init__`. Add `link_contentions: int` and `total_link_wait_cycles: int` to `SimResult`. Remove any `color_contentions` reference.

**`benchmark.py`**: Update metric names (`color_contentions` → `link_contentions`, add `total_link_wait_cycles`).

### Test updates

- Remove all references to `hop_latency` in test setup. Known locations:
  - `tests/python/runtime/test_bridge.py` (MeshConfig construction)
  - `tests/python/compiler/passes/test_lower.py` (artifact config)
  - `tests/python/compiler/test_artifact.py` (artifact config)
  - `crates/mesh_runtime/src/program.rs` (inline test JSON strings)
- Rename `color_contentions` references to `link_contentions`
- All tests should compile and pass with unchanged timing (Phase 0 changes only data model, not timing logic)

---

## Phase 1: Wormhole Timing in `forward_message`

### Current behavior

```rust
fn forward_message(&mut self, timestamp: u64, coord: Coord, message: &mut Message, dir: Direction) {
    // ... neighbor calculation, profiling ...
    let key = (coord, neighbor, message.color);  // per-(link, color)
    let free_at = self.link_color_free_at.get(&key).copied().unwrap_or(0);
    let entry_time = std::cmp::max(timestamp, free_at);
    self.link_color_free_at.insert(key, entry_time + self.config.hop_latency);  // +1 tick
    // ... create forwarded message ...
    self.enqueue_deliver(entry_time + self.config.hop_latency, neighbor, coord, new_message);  // arrive after 1 tick
}
```

### New behavior

```rust
fn forward_message(&mut self, timestamp: u64, coord: Coord, message: &mut Message, dir: Direction) {
    // ... neighbor calculation, profiling ...
    let key = (coord, neighbor);  // per-link (color-independent)
    let free_at = self.link_free_at.get(&key).copied().unwrap_or(0);
    let entry_time = std::cmp::max(timestamp, free_at);

    // Track contention
    if entry_time > timestamp {
        self.profile.link_contentions += 1;
        self.profile.total_link_wait_cycles += entry_time - timestamp;
    }

    let payload_len = message.payload.len() as u64;
    let occupancy = std::cmp::max(payload_len, 1);  // minimum 1 cycle even for empty messages

    // Link occupied for payload.len() cycles (all flits pass through)
    self.link_free_at.insert(key, entry_time + occupancy);

    // Head flit arrives at neighbor after 1 cycle
    let head_flit_arrival = entry_time + 1;

    // ... create forwarded message ...
    self.enqueue_deliver(head_flit_arrival, neighbor, coord, new_message);
}
```

Key changes:
- Link key drops color: `(coord, neighbor)` instead of `(coord, neighbor, color)`
- Link occupancy = `payload.len()` cycles (not 1)
- Head flit still arrives after 1 cycle (wormhole pipelining)
- Contention tracking counts wait cycles

### Timing example

Message with 100 elements, 3 hops, no contention:
- Link 0: entry t=0, occupied until t=100. Head arrives at hop 1 at t=1.
- Link 1: entry t=1, occupied until t=101. Head arrives at hop 2 at t=2.
- Link 2: entry t=2, occupied until t=102. Head arrives at destination at t=3.
- Payload fully available at destination: t=3 + 100 - 1 = t=102 (Phase 2 handles this).
- End-to-end: 100 + 3 - 1 = 102 cycles.

---

## Phase 2: Tail-Flit Delivery Delay

### Final destination delivery

Currently, `process_deliver` at the final destination triggers tasks at `timestamp + task_base_latency`. With wormhole routing, the head flit arrives first but the full payload isn't available until `payload.len() - 1` cycles later.

```rust
// In process_deliver, final destination branch:
let tail_flit_delay = if payload.len() > 1 { (payload.len() as u64) - 1 } else { 0 };
let ready_time = timestamp + tail_flit_delay;

pe.write_slot(payload_slot, payload);
// ... trigger tasks at ready_time + task_base_latency ...
```

### DeliverAndForward

For intermediate PEs with DeliverAndForward routing entries, two things happen:
1. **Head flit forwarding**: The head flit is forwarded immediately at `timestamp` (no tail-flit wait — wormhole forwarding starts on head flit).
2. **SRAM copy**: The payload copy is written to the deliver_slot after the tail flit passes through, at `timestamp + payload.len() - 1`.

```rust
RouteAction::DeliverAndForward { direction, deliver_slot } => {
    // Forward head flit immediately (wormhole)
    self.forward_message(timestamp, coord, message, direction);

    // SRAM delivery waits for tail flit
    let tail_flit_delay = if message.payload.len() > 1 {
        (message.payload.len() as u64) - 1
    } else {
        0
    };
    let ready_time = timestamp + tail_flit_delay;
    pe.write_slot(deliver_slot, message.payload.clone());
    // ... trigger tasks at ready_time + task_base_latency ...
}
```

### Edge cases

- **Empty payload (len=0)**: No tail-flit delay. Tasks trigger immediately (as before).
- **Single element (len=1)**: No tail-flit delay (head flit = tail flit).

---

## Phase 3: Remove Broadcast/Scatter Serialization

### Current behavior

`broadcast_to_dests` serializes same-color sends by `+1` tick:

```rust
let mut color_counts: HashMap<u32, u64> = HashMap::new();
for route in routes {
    let count = color_counts.entry(route.color).or_insert(0);
    let send_time = base_time + *count;
    *count += 1;
    self.emit_message(send_time, ...);
}
```

### New behavior

All sends emitted at `base_time`. Link contention via `link_free_at` naturally serializes messages sharing the same outgoing link:

```rust
for route in routes {
    self.emit_message(base_time, coord, route.dest, payload.clone(), route.payload_slot, route.color);
}
```

If two messages share the same first-hop link (common for XY routing from the same PE), the first message (by event processing order) occupies the link for `payload.len()` cycles. The second message's head flit waits until the link is free.

If two messages go to different first-hop neighbors (different outgoing links), they depart in parallel — no contention.

Apply the same change to `scatter_to_dests`.

### Impact

This is where M10's "parallel sends across colors" model is corrected. Colors no longer provide per-link bandwidth parallelism. The throughput advantage of color diversity now comes from routes diverging to different physical links, not from multiplexing on the same link.

---

## Phase 4: Re-baseline Benchmarks and Profiling

### Expected timing impact

Timestamps will increase dramatically. A message carrying 100 elements across 3 hops now takes ~102 cycles instead of 3. The absolute numbers change but the relative ordering of optimization effects (placement, routing, tiling) remains — and becomes more meaningful because payload size is now visible.

### Test updates

- **Rust unit tests**: Update expected timestamps in all timing-dependent tests. Focus on verifying the wormhole formula: `payload.len() + hops - 1`.
- **Python end-to-end tests**: Numerical outputs must be identical (only timing changes, not computation). Update expected `final_timestamp` values.
- **Benchmark regression tests**: Re-baseline all threshold assertions with new timing values.

### Profiling updates

**`benchmark.py`** — report new metrics:
- `link_contentions` (replaces `color_contentions`)
- `total_link_wait_cycles` (new: cumulative cycles messages spent waiting)

### Verification checklist

- All numerical outputs identical to pre-M11 (same activations, same weights, same results)
- Timing increases proportional to payload sizes
- `link_contentions` = 0 for simple single-message tests
- Multi-message tests show correct contention behavior
- Wormhole formula holds: single message, N elements, H hops → `N + H - 1` cycles end-to-end

---

## Out of scope (deferred)

- **Backpressure / flow control**: Finite input queue capacity and wormhole stall propagation. Natural M12 candidate.
- **Configurable fabric width**: Currently hardcoded 1 word/cycle. Could add `fabric_words_per_cycle` parameter later.
- **fp16 computation**: Runtime still computes in f32. Only the fabric timing model uses element-count-as-word-count.
- **Store-and-forward comparison mode**: Could be useful for benchmarking wormhole vs SAF, but not needed now.
