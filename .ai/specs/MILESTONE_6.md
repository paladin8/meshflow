# Milestone 6: Observability and Profiling

## Objective

Make the system explain itself. Enrich the Rust runtime with granular profiling data — per-operator latency, per-link message counts, queue depth tracking, and an event timeline — then provide Python matplotlib helpers that turn that data into static PNG visualizations. No HTTP API changes; the viz tools work directly against `SimResult` as a Python library.

## Exit criteria

1. Rust `ProfileSummary` includes per-operator timing, per-link message counts, per-PE max queue depth, and an event timeline log.
2. `SimResult` (PyO3) exposes all new profiling data to Python.
3. Six matplotlib plot functions in `python/meshflow/viz/` produce PNG files:
   - PE utilization heatmap
   - SRAM usage bar chart
   - Route contention mesh overlay
   - Per-operator latency bars
   - Queue depth histogram
   - Event timeline scatter
4. A `dump_all()` convenience function generates all six plots.
5. Python tests verify each plot function produces a non-empty PNG.
6. Rust tests verify new counters and data structures are populated correctly.
7. All existing tests continue to pass.
8. All linters clean (`cargo fmt`, `clippy`, `ruff`, `mypy`).
9. `matplotlib` added to `pyproject.toml` dependencies.

---

## 1. Rust profiling enrichment

All changes are additive to existing profiling infrastructure. Existing fields on `PeCounters`, `ProfileSummary`, and `SimResult` are unchanged.

### 1.1 New data structures in `profiling.rs`

**`TraceEvent`** — one entry per event processed by the simulator:

```rust
pub struct TraceEvent {
    pub timestamp: u64,
    pub coord: Coord,
    pub kind: TraceEventKind,
    pub detail: String,
}

pub enum TraceEventKind {
    MessageDeliver,
    TaskExecute,
    MessageSend,
}
```

The `detail` field carries context: the task kind name for `TaskExecute`, or the destination coordinate for `MessageSend`/`MessageDeliver`.

**Task kind string mapping** — `TaskKind` needs a `Display` impl (or equivalent helper) that produces these strings, matching the serde tag names used in `program.rs`:

| `TaskKind` variant | String |
|---------------------|--------|
| `ForwardActivation` | `"forward_activation"` |
| `CollectOutput` | `"collect_output"` |
| `Linear` | `"linear"` |
| `ConcatCollect` | `"concat_collect"` |
| `ConcatCollectForward` | `"concat_collect_forward"` |

These strings are used in `OperatorTiming.task_kind`, `TraceEvent.detail`, and the Python-side `operator_timings` dicts.

**`OperatorTiming`** — recorded each time a task executes:

```rust
pub struct OperatorTiming {
    pub task_kind: String,
    pub coord: Coord,
    pub start_ts: u64,
    pub end_ts: u64,
}
```

`start_ts` is the `ExecuteTask` event timestamp (the time at which the task actually runs — the scheduler has already added `task_base_latency` before this point). `end_ts` is `start_ts + task_base_latency`, representing a modeled duration for visualization purposes. In the current runtime, all tasks have the same `task_base_latency`, so this chart effectively shows invocation counts weighted by latency. These group naturally by `task_kind` for per-operator latency analysis.

**Per-link message counts** — added to `ProfileSummary`:

```rust
pub link_counts: HashMap<(Coord, Coord), u64>,
```

Keyed by directed edge `(from_pe, to_pe)`. Incremented each time a message hops between two adjacent PEs.

### 1.2 Changes to existing structures

**`PeCounters`** gains one field:

```rust
pub max_queue_depth: u64,
```

This tracks the peak number of pending `DeliverMessage` events targeting a given PE in the global `EventQueue` at any point during the simulation. The runtime does not use per-PE input queues — all events flow through the global `EventQueue`. To track this, the `Simulator` maintains two auxiliary maps:

- `pending_counts: HashMap<Coord, u64>` — current pending event count per PE, incremented on enqueue and decremented on dequeue.
- `max_pending: HashMap<Coord, u64>` — peak pending count per PE, updated whenever `pending_counts[coord]` exceeds the current max.

At the end of the simulation (before building `ProfileSummary`), the tracked `max_pending` values are written into each PE's `PeCounters.max_queue_depth` field. This happens before the existing `profile.per_pe.insert(pe.coord, pe.counters.clone())` step in `build_summary()`.

**`ProfileSummary`** gains three fields:

```rust
pub trace_events: Vec<TraceEvent>,
pub operator_timings: Vec<OperatorTiming>,
pub link_counts: HashMap<(Coord, Coord), u64>,
```

### 1.3 Instrumentation in `runtime.rs`

The event loop is modified at four points:

1. **Message enqueue** — when a `DeliverMessage` event is pushed to the global `EventQueue`, increment the pending count for the target PE (`pending_counts[coord] += 1`). If the new count exceeds `max_queue_depth` for that PE, update it. When a `DeliverMessage` event is popped for processing, decrement the pending count (`pending_counts[coord] -= 1`).
2. **Message hop** — in the intermediate-hop branch of `process_deliver` (where a message has not yet reached its destination), increment `link_counts[(coord, neighbor)]` where `coord` is the current PE and `neighbor` is the next hop PE.
3. **Task execution** — record an `OperatorTiming` with the task's kind string and the current timestamp. Also push a `TraceEvent` with `kind: TaskExecute`.
4. **Event processing** — push a `TraceEvent` with `kind: MessageDeliver` when processing a `DeliverMessage` event. Push a `TraceEvent` with `kind: MessageSend` at message-enqueue sites (inside `process_execute` and the intermediate-hop branch of `process_deliver`) — this does not correspond to an `EventKind` variant; it's emitted when a new `DeliverMessage` is pushed to the queue.

### 1.4 Changes to `bridge.rs`

Expose new data on the existing `SimResult` PyO3 class and `PeStats` PyO3 class:

**`PeStats`** gains:

- `max_queue_depth: u64` — `#[pyo3(get)]`

**`SimResult`** gains:

- `trace_events` — list of dicts, each with keys `timestamp` (int), `coord` (tuple), `kind` (str: `"message_deliver"`, `"task_execute"`, `"message_send"`), `detail` (str).
- `operator_timings` — list of dicts, each with keys `task_kind` (str), `coord` (tuple), `start_ts` (int), `end_ts` (int).
- `link_counts` — dict mapping `((x1, y1), (x2, y2))` tuples to int counts.

All new fields are read-only properties populated during `sim_result_to_py()` conversion. The `Coord` pairs in `link_counts` are manually converted to nested tuples `((x1, y1), (x2, y2))`, consistent with the existing pattern for `pe_stats` and `outputs` coordinate conversion.

---

## 2. Python viz module

### 2.1 File layout

| File | Responsibility |
|------|---------------|
| `python/meshflow/viz/__init__.py` | Re-exports plot functions and `dump_all` |
| `python/meshflow/viz/heatmap.py` | `pe_heatmap()` |
| `python/meshflow/viz/sram.py` | `sram_usage()` |
| `python/meshflow/viz/contention.py` | `route_contention()` |
| `python/meshflow/viz/latency.py` | `operator_latency()` |
| `python/meshflow/viz/queue.py` | `queue_depth()` |
| `python/meshflow/viz/timeline.py` | `event_timeline()` |
| `python/meshflow/viz/dump.py` | `dump_all()` convenience |
| `tests/python/viz/test_viz.py` | Tests for all plot functions |

### 2.2 Function signatures

Each function takes a `SimResult` (and optionally a `RuntimeProgram` or mesh dimensions where needed) and writes a PNG to the specified path.

```python
def pe_heatmap(
    result: SimResult,
    mesh_width: int,
    mesh_height: int,
    metric: str = "tasks_executed",
    output_path: Path = Path("artifacts/traces/pe_heatmap.png"),
) -> Path:
    """2D grid colored by a per-PE metric.

    `metric` is one of: "messages_received", "messages_sent",
    "tasks_executed", "slots_written", "max_queue_depth".
    """

def sram_usage(
    program: RuntimeProgram,
    output_path: Path = Path("artifacts/traces/sram_usage.png"),
) -> Path:
    """Bar chart of initial SRAM slot count per PE."""

def route_contention(
    result: SimResult,
    mesh_width: int,
    mesh_height: int,
    output_path: Path = Path("artifacts/traces/route_contention.png"),
) -> Path:
    """Mesh grid with edges colored by link message counts."""

def operator_latency(
    result: SimResult,
    output_path: Path = Path("artifacts/traces/operator_latency.png"),
) -> Path:
    """Bar chart of total time per task kind."""

def queue_depth(
    result: SimResult,
    output_path: Path = Path("artifacts/traces/queue_depth.png"),
) -> Path:
    """Histogram of max queue depth across PEs."""

def event_timeline(
    result: SimResult,
    output_path: Path = Path("artifacts/traces/event_timeline.png"),
) -> Path:
    """Scatter plot: logical time (X) vs PE coord (Y), colored by event kind."""

def dump_all(
    result: SimResult,
    program: RuntimeProgram,
    output_dir: Path = Path("artifacts/traces"),
) -> list[Path]:
    """Generate all six plots into output_dir.

    Extracts mesh_width and mesh_height from program.mesh_config.
    Returns list of output paths.
    """
```

All functions return the output path for composability. Parent directories are created if needed.

Functions that need `mesh_width` and `mesh_height` require the caller to pass them explicitly. These are available from `program.mesh_config.width` and `program.mesh_config.height` on the Python `RuntimeProgram` dataclass. The `dump_all` convenience function takes both `result` and `program` and extracts dimensions automatically.

### 2.3 Plot details

**PE heatmap:** Uses `matplotlib.pyplot.imshow` on a `(height, width)` grid. Color scale is linear. Title and colorbar label reflect the selected metric.

**SRAM usage:** Horizontal bar chart with one bar per PE (labeled by coordinate). Bar length is the total number of floats across all SRAM slots: `sum(len(v) for v in pe.initial_sram.values())`. This measures memory pressure more usefully than raw slot count. Derived from the `RuntimeProgram` (specifically `program.pe_programs[].initial_sram`), not from `SimResult`.

**Route contention:** Draws PE nodes as circles on a grid. Draws edges between adjacent PEs as lines, colored and thickened by `link_counts` value. Uses a sequential colormap (e.g., `YlOrRd`). Edges with zero traffic are drawn thin and gray.

**Operator latency:** Groups `operator_timings` by `task_kind`, sums `(end_ts - start_ts)` per group. Vertical bar chart, one bar per task kind. Note: since the simulator uses a uniform `task_base_latency` (default 1), this chart effectively shows invocation counts weighted by latency. Different task kinds may get distinct latencies in a future milestone.

**Queue depth:** Histogram of `max_queue_depth` values across all PEs. X axis is depth, Y axis is PE count. Shows distribution of buffering pressure.

**Event timeline:** Scatter plot. X = `timestamp`, Y = PE index (linearized from coord, e.g., `y * width + x`). Color = event kind (three colors for deliver/execute/send). Shows execution ordering and idle gaps.

---

## 3. Dependencies

Add `matplotlib` to `pyproject.toml` under `[project] dependencies`.

No other new dependencies.

---

## 4. Testing

### 4.1 Rust tests

In `crates/mesh_runtime/src/` (unit tests within modules or in `tests/`):

- **`max_queue_depth` tracking:** Run a small mesh scenario with multiple messages targeting the same PE. Verify the `max_queue_depth` counter on that PE reflects the peak number of pending events.
- **`link_counts` tracking:** Run a small mesh scenario where a message traverses multiple hops. Verify link counts are `1` for each hop edge.
- **`operator_timings` recording:** Run a mesh with at least one task. Verify an `OperatorTiming` entry exists with correct `task_kind`, `coord`, and `start_ts < end_ts`.
- **`trace_events` recording:** Run a small mesh scenario. Verify trace events are produced for both message deliveries and task executions.
- **Existing tests pass** — no regressions.

### 4.2 Python tests

In `tests/python/viz/test_viz.py`:

- Compile and run a LINEAR-based graph (e.g., single-layer MLP with tiling) to exercise richer profiling data — multiple task kinds, multi-hop messages, fragment collection. Call each of the six plot functions with `tmp_path` output paths.
- Assert each output file exists and has size > 0.
- A secondary minimal test with FORWARD→COLLECT verifies the plots handle simple/sparse data gracefully.
- Test `dump_all` produces all six files.
- No pixel-level assertions — matplotlib output is inherently visual.

Uses `matplotlib.use("Agg")` backend so tests run headless.

---

## 5. Scope boundaries

**In scope:**

- Rust profiling enrichment: per-operator timing, per-link counts, max queue depth, event timeline.
- PyO3 exposure of all new data.
- Six matplotlib plot functions + `dump_all`.
- Rust and Python tests.
- `matplotlib` dependency.

**Out of scope (deferred):**

- Step-by-step trace replay / interactive debugger.
- HTTP API changes — detailed profiling is library-only.
- Interactive / HTML visualizations.
- Real-time streaming of profiling data.
- Automatic reference-model comparison in the viz layer.
- PE SRAM capacity enforcement in the runtime.
