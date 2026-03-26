# Meshflow

## Design Document

## 1. Summary

Meshflow is a Cerebras-inspired spatial inference engine designed to teach the core ideas behind low-latency inference on a spatially programmed mesh of processing elements (PEs):

* model weights are preloaded into PE-local memory,
* activations flow through a statically placed dataflow pipeline,
* execution is driven by explicit placement, routing, and scheduling,
* an inference API exposes the system end to end.

The implementation will use:

* **Python** for model import, graph lowering, compiler passes, orchestration, and the API layer,
* **Rust** for the mesh simulator core, routing/runtime data structures, and performance-sensitive execution.

The goal is not to reproduce Cerebras faithfully. The goal is to build a compact, educational system that makes the key architectural ideas concrete.

---

## 2. Goals

### 2.1 Primary goals

Build a working end-to-end system with the following properties:

1. A client can send an inference request through a simple HTTP API.
2. A small neural network is compiled into a **spatial execution plan**.
3. The plan is loaded into a simulated 2D PE mesh.
4. Model weights are stored in PE-local memory before execution.
5. Activations propagate through the mesh according to explicit routes.
6. The simulator produces numerically correct outputs relative to a Python reference implementation.
7. The system exposes enough tracing and profiling to understand latency, utilization, routing cost, and memory pressure.

### 2.2 Secondary goals

* Make the compiler architecture clean enough to support later experimentation.
* Make the simulator realistic enough to surface architectural tradeoffs.
* Keep the initial model small enough that the project actually ships.
* Keep the Python/Rust boundary narrow and intentional.

---

## 3. Non-goals

This project is explicitly **not** trying to do the following in v1:

* reproduce the real Cerebras software stack,
* implement a full transformer serving system,
* build a production-grade continuous batching runtime,
* support arbitrary PyTorch models,
* implement cycle-accurate hardware timing,
* build a full LLVM backend,
* match GPU-level throughput,
* support distributed multi-host mesh execution.

If the project succeeds, it should feel like a serious educational artifact, not a benchmark contender.

---

## 4. High-level architecture

The system has four major layers:

1. **Inference API / Orchestration Layer (Python)**

   * Loads a compiled model artifact.
   * Accepts inference requests.
   * Marshals input tensors to the simulator.
   * Returns outputs and optional traces.

2. **Compiler Layer (Python)**

   * Imports a small model description.
   * Lowers it into a graph IR.
   * Partitions the graph spatially across the mesh.
   * Generates explicit routing and scheduling metadata.
   * Emits a compiled artifact consumable by the Rust runtime.

3. **Runtime / Simulator Core (Rust)**

   * Represents the PE mesh.
   * Owns PE-local memory, queues, and execution state.
   * Executes activation flow across the mesh.
   * Collects timing and profiling statistics.

4. **Reference / Validation Layer (Python)**

   * Runs the same model with a plain Python or PyTorch implementation.
   * Compares outputs against simulator results.
   * Supports test-driven development of compiler and runtime components.

### 4.1 Conceptual execution model

Execution in v1 will use the following mental model:

* The mesh is divided into regions.
* Each region is responsible for one operator or one operator fragment.
* Weights for that operator are preloaded into the local memory of the region's PEs.
* Inputs enter the mesh at source PEs.
* Activations move between regions along explicit routes.
* Each PE runs a small task machine over local state and incoming messages.
* Outputs are collected from designated sink PEs.

This is a **resident-weights, activation-flow** design.

---

## 5. Model scope

### 5.1 MLP pipeline (complete)

The first supported model was a small feed-forward network:

```text
Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
```

This established: matrix-vector compute with row-tiling, explicit activation routing between layers, fragment collection, and straightforward numerical validation.

### 5.2 Single-head transformer block (complete)

After the MLP pipeline stabilized, a single-head transformer block was implemented:

```text
Input → RMSNorm → Q/K/V proj → attention (QK^T → Softmax → AV) → out_proj → Add → RMSNorm → FFN → Add → Output
```

This added: RMSNorm (two-phase reduce-broadcast), MatMul (matrix-vector multiply), Softmax (numerically stable row-wise), Add (element-wise residual), skip connections, scatter/broadcast routing, and multi-position (seq_len > 1) support. See `.ai/specs/MILESTONE_7.md` for full details.

---

## 6. Core abstractions

### 6.1 Processing element (PE)

Each PE in the simulated mesh contains:

* a unique `(col, row)` coordinate,
* a slot-addressed local SRAM (named `f32` vector slots),
* an input queue for arriving messages,
* a set of configured tasks (each with a trigger slot),
* an optional SRAM capacity limit in bytes,
* profiling counters (tasks executed, messages sent/received, hops).

A PE is not a scalar CPU core. It is a configurable execution cell with local memory and message-driven computation.

### 6.2 Message

A message represents an activation payload moving through the mesh.

Message fields:

* `id` — unique message identifier,
* `source` / `dest` — PE coordinates,
* `hops` — pre-computed hop list (sequence of N/S/E/W directions),
* `current_hop` — progress index into the hop list,
* `payload` — inline `Vec<f32>` data,
* `payload_slot` — destination SRAM slot on the receiving PE,
* `timestamp` — logical time for the simulator.

### 6.3 Task

A task is a unit of computation triggered when a PE has the required local state and inputs. Tasks are configured statically by the compiler via `TaskConfig` entries, each specifying a `TaskKind` and a `trigger_slot`.

Current `TaskKind` variants:

* `ForwardActivation` — read input and route to a destination PE,
* `CollectOutput` — mark input as simulation output,
* `Linear` — compute `W @ x + b` for a weight tile, emit output fragment,
* `ConcatCollect` — accumulate fragments into a single output vector,
* `ConcatCollectForward` — accumulate fragments, optionally apply activation, broadcast/scatter onward,
* `Add` — element-wise addition of two SRAM slots (dual-trigger with has_slot guard),
* `Softmax` — numerically stable row-wise softmax (in-place on a single PE),
* `MatMul` — matrix-vector multiply with optional transpose (dual-trigger with has_slot guard),
* `RmsNormPartialSum` — compute local `sum(x^2)` slice, send to reduce PE,
* `RmsNormReduce` — accumulate partial sums, compute scale factor, broadcast,
* `RmsNormNormalize` — apply `x * scale * gamma` per position.

### 6.4 Operator groups

An operator group is a set of PEs assigned to one operator or one operator partition. The compiler's expand pass decomposes GraphIR nodes into groups:

* `TiledComputeGroup` — LINEAR operators: tile PEs (each holding a weight tile) + a collect PE,
* `RmsNormGroup` — tile PEs + reduce PE + collect PE,
* `AttentionGroup` — `seq_len` attention PEs (co-located QK^T MatMul + Softmax + AV MatMul) + optional collect PE,
* `PassthroughGroup` — single PE for ADD, standalone SOFTMAX, or other non-tiled ops.

Groups are placed column-per-group on the mesh, with tile PEs arranged vertically.

### 6.5 Compiled artifact

The compiler output is a versioned msgpack-serialized `RuntimeProgram` containing:

* `version` — artifact format version (currently 1),
* `mesh` — mesh dimensions (width × height),
* `pes` — per-PE programs, each with:
  * task programs (one per configured task),
  * initial SRAM contents (pre-loaded weights and biases),
* `inputs` — input slot bindings (which PE/slot receives external input).

---

## 7. Intermediate representations

Custom IRs are used throughout. No LLVM.

### 7.1 Graph IR

Represents the model at an operator graph level. Defined in `compiler/graph_ir.py`.

Node types (`OpType`): `FORWARD`, `COLLECT`, `LINEAR`, `RELU`, `ADD`, `SOFTMAX`, `RMSNORM`, `MATMUL`.

Each node has an `id`, an `op` type, and optional `attrs` (e.g., `in_features`, `out_features` for LINEAR; `eps`, `feature_count` for RMSNORM; `seq_len`, `d_model` for MATMUL).

Edges connect `(src_node, src_slot)` → `(dst_node, dst_slot)`. Slot indices determine SRAM placement on destination PEs.

Purpose: correctness-oriented model representation, input to compiler passes. Includes DAG validation and per-op-type attribute/connectivity checks.

### 7.2 Expanded IR

Represents the graph after operator decomposition. Defined in `compiler/expanded_ir.py`.

Each GraphIR node is expanded into an operator group (see Section 6.4). The expansion pass determines tile counts, creates internal PEs (tile, reduce, collect), and records `NodeExpansion` mappings (input_pe_ids, output_pe_ids) for inter-group edge generation.

Questions answered here: how many tiles per operator, which PEs are tile/reduce/collect, what internal edges exist within each group.

### 7.3 Spatial IR

Represents the graph after physical placement. Defined in `compiler/spatial_ir.py`.

Each expanded node is assigned a `(col, row)` coordinate. Groups are laid out column-per-group with tile PEs vertically. Internal edges (tile→reduce, reduce→tile, tile→collect) and inter-group edges are generated. `PlacedNodeKind` provides clean dispatch: FORWARD, COLLECT_SIMPLE, LINEAR_TILE, LINEAR_COLLECT, RMSNORM_TILE, RMSNORM_REDUCE, ATTENTION_PE, ADD, SOFTMAX.

Mesh dimensions are derived from the placement result.

### 7.4 Schedule IR

Represents per-PE task configurations and routing. Defined in `compiler/schedule_ir.py`.

The route pass dispatches on `PlacedNodeKind` to generate task entries (one per PE task), SRAM pre-loads (weights, biases, gamma), and XY hop lists for inter-PE communication. Contains `TaskEntry` variants matching all `TaskKind` variants in the runtime.

### 7.5 Runtime Program

The final low-level form consumed by Rust. Defined in `compiler/artifact.py`, loaded in `program.rs`.

Contains: per-PE task programs, initial SRAM contents, input slot bindings, mesh dimensions, and a version field. Serialized as msgpack via `serialize()` / `deserialize()`.

---

## 8. Compiler design

### 8.1 Compiler pipeline

The compiler has four passes, invoked via `compile(graph, config, weights)`:

1. **Expand** (`passes/expand.py`)

   * Decompose each GraphIR node into an operator group (tiled, RmsNorm, attention, or passthrough).
   * Determine tile counts based on `mesh_height` and reserved rows.
   * Detect attention chains (MATMUL → SOFTMAX → MATMUL) and fuse them into a single AttentionGroup.
   * Produce Expanded IR with `NodeExpansion` mappings.

2. **Place** (`passes/place.py`)

   * Assign each expanded node a `(col, row)` coordinate on the mesh.
   * Column-per-group layout: each operator group gets its own column, tile PEs arranged vertically.
   * Generate internal edges (tile→reduce, reduce→tile, tile→collect, attn→collect) and inter-group edges.
   * Derive mesh dimensions from the placement result.
   * Produce Spatial IR.

3. **Route** (`passes/route.py`)

   * For each placed node, generate task entries, SRAM pre-loads, and XY hop lists.
   * Dispatch on `PlacedNodeKind` for operator-specific routing logic.
   * Handle broadcast (one source → N destinations) and scatter (row i → destination i) patterns.
   * Split weights into tiles and assign to PE SRAM slots.
   * Produce Schedule IR.

4. **Lower** (`passes/lower.py`)

   * Mechanical translation from Schedule IR task entries to Runtime Program task programs.
   * Emit per-PE programs with task configs and initial SRAM.
   * Serialize as msgpack artifact.

### 8.2 Placement strategy

Column-per-group layout:

* each operator group gets its own column of PEs,
* tile PEs are arranged vertically within the column,
* collect and reduce PEs are placed below the tiles,
* mesh dimensions grow to accommodate the widest group,
* for mixed graphs (not passthrough-only), columns are assigned left to right.

### 8.3 Weight layout strategy

For a `Linear(in_dim, out_dim)` operator:

* partition weights by output rows: `base = out_dim // num_tiles`, first `remainder` tiles get `base + 1` rows,
* each tile PE holds its weight tile and bias slice in SRAM (slots 1, 2),
* input activation is broadcast to all tile PEs (arrives at slot 0),
* each tile computes its partial output and routes the fragment to the collect PE.

### 8.4 Routing strategy

Deterministic XY dimension-ordered routing on a 2D Manhattan mesh:

* messages first move horizontally (East/West), then vertically (North/South),
* pre-computed hop lists stored in each task (no runtime route computation),
* single message class, no adaptive routing,
* explicit hop counting for profiling.

### 8.5 Scheduling strategy

Message-driven readiness:

* each task has a `trigger_slot` — the task fires when a write to that SRAM slot occurs,
* single-trigger tasks (e.g., Linear, ForwardActivation) execute immediately on trigger,
* dual-trigger tasks (e.g., Add, MatMul) use a `has_slot` guard — two TaskConfig entries, each triggers on one input, but computation only runs when both inputs are present,
* counter-based tasks (e.g., ConcatCollect, RmsNormReduce) accumulate fragments and only compute when all have arrived,
* tasks consume input slots after computation to prevent re-triggering.

---

## 9. Simulator design

### 9.1 Simulator scope

The simulator should be **event-driven**, not cycle-accurate.

The runtime models:

* PE-local memory,
* message queues,
* task execution,
* route traversal,
* coarse timing costs.

The runtime does not model:

* wires at hardware fidelity,
* real SRAM banking,
* exact network arbitration,
* exact instruction scheduling.

### 9.2 Timing model

Use a coarse cost model with configurable costs such as:

* `hop_latency`
* `task_base_latency`
* `matvec_cost_per_element`
* `queue_push_cost`
* `queue_pop_cost`
* `reduction_cost`

The objective is not hardware accuracy. The objective is comparative intuition.

### 9.3 Execution loop

A rough event loop:

1. Seed input messages into source PEs.
2. Pop the next runnable event.
3. Deliver message or execute task.
4. Update time and profiling counters.
5. Enqueue resulting events.
6. Repeat until sink outputs are complete.

### 9.4 Rust responsibilities

Rust should own:

* mesh data structures,
* event queues,
* PE-local memory representation,
* route traversal,
* task dispatch,
* profiling collection,
* compiled program loading.

### 9.5 Python responsibilities

Python should not own the inner simulation loop.

Python should:

* call into Rust with a compiled artifact,
* provide input tensors,
* receive outputs and traces,
* present debugging and visualization helpers.

---

## 10. Python/Rust boundary

This boundary matters. Keep it narrow.

### 10.1 Boundary rule

Python performs compilation and orchestration.
Rust performs execution.

### 10.2 Crossing data structures

Data passed from Python to Rust:

* mesh config,
* compiled program artifact,
* weight blobs,
* input tensor payloads,
* simulator config.

Data returned from Rust to Python:

* output tensors,
* execution summary,
* optional trace events,
* profiling counters.

### 10.3 Binding approach

Use **PyO3** + **maturin** to expose the Rust core as a Python extension module.

This gives:

* ergonomic Python calls,
* a standard Cargo build path for Rust,
* a clean local development loop.

---

## 11. API design

### 11.1 API scope

Five endpoints implemented in FastAPI (`api/server.py`):

* `POST /compile` — compile a GraphIR + weights into a stored artifact. Accepts `graph_ir` (JSON), `weights` (nested float arrays), and `config` (mesh_height, etc.).
* `POST /run/{artifact_id}` — run a compiled artifact with input tensors. Accepts `inputs: dict[str, list[float]]`. Returns output tensors and profiling summary.
* `GET /artifacts` — list all stored artifacts.
* `DELETE /artifacts/{artifact_id}` — delete a stored artifact.
* `GET /health` — health check.

Artifacts are stored on disk via `ArtifactStore` (`api/store.py`), which manages msgpack files in a configurable directory.

### 11.2 Example payload

```json
POST /run/abc123
{
  "inputs": {"input": [0.1, 0.2, 0.3, 0.4]}
}
```

### 11.3 API implementation

FastAPI in Python, with Pydantic request/response models (`api/schemas.py`).

### 11.4 Future enhancements (not yet implemented)

* Optional `trace` flag on `/run` to return trace events in the response.
* Raw trace export to JSON/CSV files for offline analysis.

---

## 12. Observability and debugging

### 12.1 Implemented diagnostics

All of the following are implemented:

* **Mesh layout dump** — `viz/dump.py`: ASCII mesh state visualization.
* **SRAM usage by PE** — `viz/sram.py`: horizontal bar chart of per-PE float counts.
* **PE utilization heatmap** — `viz/heatmap.py`: 2D heatmap colored by metric (tasks executed, messages sent, etc.).
* **Route contention** — `viz/contention.py`: colored edges on 2D mesh grid showing message counts per link.
* **Event timeline** — `viz/timeline.py`: scatter plot of events over logical time (X = timestamp, Y = PE index).
* **Operator latency breakdown** — `viz/latency.py`: per-operator timing visualization.
* **Queue depth analysis** — `viz/queue.py`: queue depth over time.
* **Profiling counters** — per-PE counters for tasks executed, messages sent/received, hops. Exposed via PyO3 bridge.
* **Trace events** — structured trace events collected during simulation, accessible via `SimResult.trace_events`.
* **Numerical correctness** — reference implementations in `models/reference.py` for all supported operators.

All visualization functions export to `artifacts/traces/` as PNG files.

### 12.2 Future enhancements (not yet implemented)

* Step-by-step trace replay for interactive debugging.
* Raw trace export to JSON/CSV files.
* Golden/snapshot artifact tests that detect unintended compiler changes.

### 12.3 Debug philosophy

Plain data dumps and simple matplotlib visualizations. No elaborate UI. Machine-readable trace events plus Python plotting helpers.

---

## 13. Project structure

```text
meshflow/
├── pyproject.toml
├── uv.lock
├── README.md
├── rust-toolchain.toml
├── Cargo.toml
├── CLAUDE.md
├── crates/
│   └── mesh_runtime/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs                    # PyO3 module registration
│           ├── bridge.rs                 # PyO3-exposed classes + run_program()
│           ├── coords.rs                 # Coord, Direction enums
│           ├── message.rs                # Message struct
│           ├── pe.rs                     # PE, TaskConfig, TaskKind enum
│           ├── mesh.rs                   # 2D PE mesh grid
│           ├── route.rs                  # XY route generation
│           ├── event.rs                  # Event queue (min-heap)
│           ├── runtime.rs               # Simulator loop + task execution
│           ├── program.rs               # Artifact deserialization
│           ├── profiling.rs             # Counters, trace events, summaries
│           └── compiler_test_helpers.rs # Test artifact builders
├── python/
│   └── meshflow/
│       ├── __init__.py
│       ├── api/
│       │   ├── server.py                # FastAPI app + endpoints
│       │   ├── schemas.py               # Pydantic request/response models
│       │   └── store.py                 # Artifact file storage
│       ├── compiler/
│       │   ├── __init__.py              # compile() entry point
│       │   ├── config.py                # CompilerConfig
│       │   ├── graph_ir.py              # GraphIR, Node, Edge, OpType
│       │   ├── expanded_ir.py           # Operator groups + NodeExpansion
│       │   ├── spatial_ir.py            # PlacedNode, PlacedEdge, PlacedNodeKind
│       │   ├── schedule_ir.py           # Per-PE task entries + routes
│       │   ├── artifact.py              # RuntimeProgram + serialize/deserialize
│       │   └── passes/
│       │       ├── expand.py            # GraphIR → ExpandedIR
│       │       ├── place.py             # ExpandedIR → SpatialIR
│       │       ├── route.py             # SpatialIR → ScheduleIR
│       │       └── lower.py             # ScheduleIR → RuntimeProgram
│       ├── models/
│       │   ├── reference.py             # Torch reference implementations
│       │   └── transformer.py           # Transformer block model helper
│       ├── tools/
│       │   └── inspect_artifact.py      # CLI to dump .msgpack as JSON
│       └── viz/
│           ├── dump.py                  # ASCII mesh state
│           ├── heatmap.py               # PE utilization heatmaps
│           ├── sram.py                  # SRAM usage visualization
│           ├── contention.py            # Route contention analysis
│           ├── latency.py               # Operator latency breakdown
│           ├── queue.py                 # Queue depth analysis
│           └── timeline.py              # Event timeline
├── tests/
│   ├── python/
│   │   ├── compiler/
│   │   │   ├── test_graph_ir.py
│   │   │   ├── test_expanded_ir.py
│   │   │   ├── test_artifact.py
│   │   │   ├── test_compile.py
│   │   │   └── passes/
│   │   │       ├── test_expand.py
│   │   │       ├── test_place.py
│   │   │       ├── test_route.py
│   │   │       └── test_lower.py
│   │   ├── runtime/
│   │   │   ├── test_bridge.py
│   │   │   ├── test_end_to_end.py
│   │   │   ├── test_num_positions.py
│   │   │   └── test_transformer_block.py
│   │   ├── api/
│   │   │   ├── test_store.py
│   │   │   └── test_api.py
│   │   └── viz/
│   │       └── test_viz.py
│   └── rust/                            # Rust tests are inline in src files
├── artifacts/
│   ├── compiled/                        # Generated .msgpack artifacts
│   └── traces/                          # Generated visualization PNGs
├── scripts/
│   ├── test.sh
│   └── lint.sh
└── .ai/
    ├── OVERALL_DESIGN.md                # This document
    ├── specs/                           # Milestone specifications
    │   ├── MILESTONE_1.md through MILESTONE_7.md
    └── plans/                           # Implementation plans
```

Notes:

* `python/` contains the Python package. `crates/mesh_runtime/` contains the Rust simulator core.
* `artifacts/` stores generated compiled programs and trace visualizations.
* `tests/python/` uses subdirectories matching the package structure. Rust tests are inline in source files.
* `.ai/` contains design documents and milestone specs.

---

## 14. Dependency management and environment

We will use **uv** as the Python dependency and environment manager.

### 14.1 Python dependency policy

All Python dependencies should be declared in `pyproject.toml`.

Use uv for:

* creating the virtual environment,
* locking dependencies,
* installing dev dependencies,
* running Python commands and tests.

### 14.2 Rust dependency policy

Rust dependencies are managed with Cargo in the standard way.

We will not try to force Rust dependency resolution through uv. The clean split is:

* Python: uv
* Rust: cargo

### 14.3 Build integration

Use `maturin` as the bridge between Cargo and Python packaging.

Recommended approach:

* declare `maturin` as a Python dev dependency,
* use `uv run maturin develop` during local development,
* use `uv run maturin build` when producing wheels or testing packaging.

### 14.4 Python dependencies

Core (runtime):

* `fastapi`, `uvicorn`, `pydantic` — API layer
* `numpy` — tensor operations
* `orjson` — fast JSON serialization
* `msgpack` — artifact serialization
* `matplotlib` — visualization

Dev:

* `pytest`, `pytest-cov` — testing
* `ruff`, `mypy` — linting and type checking
* `maturin` — Rust extension build
* `torch` — numerical reference implementations
* `httpx` — API test client

### 14.5 Rust dependencies

* `pyo3` (0.23, extension-module) — Python bindings
* `serde` (1, derive) — serialization framework
* `rmp-serde` (1) — msgpack serialization
* `serde_json` (1) — JSON support for tests/debugging
* `thiserror` (2) — error types

The Rust dependency set is intentionally minimal.

---

## 15. Development workflow

### 15.1 Initial setup

Assuming uv, Rust, and Cargo are already installed:

```bash
uv sync
uv run maturin develop
```

This should:

* install Python dependencies,
* build the Rust extension in editable/dev form,
* make the Python package importable.

### 15.2 Running the API locally

```bash
uv run python -m meshflow.api.server
```

or

```bash
uv run uvicorn meshflow.api.server:app --reload
```

### 15.3 Inspecting a compiled artifact

```bash
uv run python -m meshflow.tools.inspect_artifact artifacts/compiled/model.msgpack
```

### 15.4 Rebuilding Rust after changes

```bash
uv run maturin develop
```

---

## 16. Testing strategy

We need layered tests. The project should not rely on ad hoc manual runs.

### 16.1 Python tests

Python tests should cover:

* Graph IR construction,
* placement logic,
* routing logic,
* artifact serialization,
* reference model correctness,
* end-to-end compilation + execution against the Rust runtime.

Run with:

```bash
uv run pytest tests/python
```

### 16.2 Rust tests

Rust tests should cover:

* route traversal,
* PE queue behavior,
* event scheduling,
* artifact loading,
* task execution correctness,
* small mesh execution scenarios.

Run with:

```bash
cargo test -p mesh_runtime
```

### 16.3 Full test pass

A full test pass should run both Python and Rust tests.

Suggested script behavior:

```bash
cargo test -p mesh_runtime
uv run pytest tests/python
```

### 16.4 Artifact tests

Round-trip serialization tests verify that artifacts survive serialize/deserialize without loss. These cover all task types and SRAM configurations.

Golden/snapshot tests (that fail on unintended compiler changes) are not yet implemented.

### 16.5 Numerical correctness tests

For every supported operator and model:

* run a reference Python/torch implementation (`models/reference.py`),
* run the compiled simulator path (`compile()` + `run_program()`),
* compare outputs within tolerance (`atol=1e-3` to `1e-4`).

Current coverage: LINEAR, ReLU, RMSNorm, Softmax, MatMul, Add, full MLP, full transformer block. All validated with multiple dimension configurations including non-divisible dimensions.

---

## 17. Coding standards

### 17.1 Python

* Format with `ruff format`.
* Lint with `ruff check`.
* Use type hints aggressively.
* Use dataclasses or Pydantic models where they make interfaces clearer.
* Prefer explicitness over metaprogramming.

### 17.2 Rust

* Format with `cargo fmt`.
* Lint with `cargo clippy`.
* Keep data structures compact and explicit.
* Prefer enums for task and message kinds.
* Avoid over-engineering generic abstractions early.

### 17.3 Shared design rule

Do not let the architecture disappear into framework cleverness.

The project should stay readable as a systems artifact.

---

## 18. Milestones

## Milestone 0: repository bootstrap — COMPLETE

Repo skeleton: `pyproject.toml` with uv, Cargo workspace with `mesh_runtime` crate, PyO3 + maturin wired up, basic package import, CI-ready test commands, README.

## Milestone 1: minimal mesh simulator — COMPLETE

Rust runtime skeleton: PE and mesh data structures, message queues, deterministic XY routing, event loop, ForwardActivation + CollectOutput tasks, profiling counters.

## Milestone 2: Graph IR and compiler skeleton — COMPLETE

Python compiler pipeline: GraphIR definitions, Expanded IR, Spatial IR, Schedule IR, artifact serialization (msgpack), round-trip tests.

## Milestone 3: single LINEAR operator — COMPLETE

Linear task in Rust with row-tiled weight distribution, ConcatCollect for fragment gathering, Python reference implementation, end-to-end compile → run → compare tests.

## Milestone 4: multi-layer MLP pipeline — COMPLETE

ReLU fusion via ConcatCollectForward, column-per-layer placement, inter-layer routing, uneven tile distribution (remainder handling), 2- and 3-layer MLP execution with torch validation.

## Milestone 5: inference API — COMPLETE

FastAPI server with 5 endpoints (compile, run, artifacts, delete, health), Pydantic schemas, ArtifactStore for disk persistence, httpx-based API tests.

## Milestone 6: observability and profiling — COMPLETE

Rust profiling enrichment (per-PE counters, trace events, operator timing), PyO3 bridge for profiling data, matplotlib visualization functions (heatmap, timeline, SRAM, contention, latency, queue depth).

## Milestone 7: single-head transformer block — COMPLETE

Six new Rust TaskKind variants (RmsNormPartialSum, RmsNormReduce, RmsNormNormalize, Softmax, MatMul, Add). Two-phase reduce-broadcast for RMSNorm. Row-parallel attention with co-located QK^T + Softmax + AV. Dual-trigger has_slot guard pattern for Add and MatMul. Skip connections via ForwardActivation payload_slot. Multi-position (seq_len > 1) support across all operators. Scatter/broadcast routing. Full end-to-end tests including non-divisible dimensions. See `.ai/specs/MILESTONE_7.md` for complete details.

---

## 19. Implementation order (completed)

The actual implementation order followed the planned sequence:

1. Repo bootstrap (M0)
2. Rust mesh and route primitives (M1)
3. Python IR definitions + artifact format (M2)
4. Linear operator execution (M3)
5. Multi-layer MLP with ReLU (M4)
6. API layer (M5)
7. Profiling and visualization (M6)
8. Transformer block operators (M7)

The first working vertical slice was kept extremely small (single message forwarding), then expanded incrementally.

---

## 20. Risks and mitigations (retrospective)

### 20.1 Simulator realism

Risk: making the simulator too realistic too early. Mitigation: coarse timing model, deterministic XY routing, event-driven (not cycle-accurate). This held — the timing model remains simple and the routing is fully deterministic.

### 20.2 Artifact boundary stability

Risk: compiler and runtime co-evolving without a stable artifact boundary. Mitigation: versioned artifact format defined early, round-trip serialization tests, msgpack format. The artifact format has been stable since M2 with backward-compatible additions via `#[serde(default)]`.

### 20.3 Python/Rust boundary

Risk: chatty and awkward boundary. Mitigation: pass whole compiled artifacts, keep the simulator loop entirely in Rust, return summary data. This held — the boundary is `compile()` (Python) → `serialize()` → `run_program()` (Rust) → `SimResult` (Python).

---

## 21. Design decisions (resolved)

These were open questions at design time, now resolved:

* **Batch size**: Started with 1, now supports `num_positions > 1` (seq_len). Multi-position support was added during M7 for attention.
* **Routes**: Explicit hop lists, pre-computed by the compiler. Stored per-task, not generated at runtime.
* **Artifact format**: msgpack (binary), not JSON. Fast serialization, compact, versioned.
* **Model definitions**: Hand-authored GraphIR via model helpers (`transformer_block()`, direct `GraphIR` construction for MLPs). No PyTorch import.
* **Tensor representation**: Dense fragments everywhere. No fragment views.

---

## 22. Definition of done for v1

All of the following are true:

* a small MLP compiles into a spatial artifact — **done**,
* the Rust simulator loads the artifact and executes it — **done**,
* weights are resident in PE-local memory — **done**,
* activations traverse explicit routes across the mesh — **done**,
* outputs match a Python reference implementation — **done** (MLP + transformer block),
* the system is callable through an HTTP API — **done**,
* traces and profiling are sufficient to explain the run — **done**,
* the repo has repeatable setup, lint, and test commands — **done**.

Additionally: a single-head transformer block (beyond the original v1 scope) compiles and executes correctly with RMSNorm, attention, FFN, and residual connections.

---

## 23. Current status

91 Rust tests, 252 Python tests, all passing. All lints clean. Milestones 0–7 complete.
