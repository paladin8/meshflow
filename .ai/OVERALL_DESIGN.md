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

## 5. Initial model scope

### 5.1 v1 target model

The initial supported model should be a small feed-forward network or MLP stack, for example:

```text
Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
```

This gives us:

* matrix-vector or matrix-batch-vector style compute,
* explicit activation movement between layers,
* straightforward validation,
* no attention or KV-cache complexity.

### 5.2 v2 target model

After the MLP pipeline works, the next target should be a tiny decoder-style block with:

* layer norm,
* a simplified attention implementation,
* an MLP block,
* residual connections.

This should only happen after the spatial pipeline and tooling are stable.

---

## 6. Core abstractions

### 6.1 Processing element (PE)

Each PE in the simulated mesh will contain:

* a unique `(x, y)` coordinate,
* a local SRAM buffer,
* an input queue or a small set of input queues,
* a set of configured tasks,
* local registers / scratch state,
* profiling counters,
* routing metadata for outbound messages.

A PE is not a scalar CPU core. It is a configurable execution cell with local memory and message-driven computation.

### 6.2 Message

A message represents an activation payload moving through the mesh.

Initial message fields:

* source PE,
* destination PE or route id,
* tensor fragment id,
* sequence / token / batch index,
* payload buffer id or inline numeric payload,
* logical timestamp for the simulator.

### 6.3 Task

A task is a unit of computation triggered when a PE has the required local state and inputs.

Examples:

* `RecvInput`
* `MatVecPartial`
* `BiasAdd`
* `ActivationRelu`
* `ReduceAccumulate`
* `ForwardActivation`
* `EmitOutput`

Tasks will be configured statically by the compiler.

### 6.4 Region

A region is a rectangular submesh assigned to one operator or one operator partition.

A region owns:

* a subset of PEs,
* a weight layout,
* an operator implementation,
* routing endpoints,
* scheduling metadata.

### 6.5 Compiled artifact

The compiler output should be a serialized artifact containing:

* model metadata,
* mesh dimensions,
* region assignments,
* task programs,
* routing tables,
* weight blobs,
* input/output bindings,
* optional debugging symbols.

Initial format can be JSON + binary blobs. We can tighten this later.

---

## 7. Intermediate representations

We will use custom IRs rather than LLVM.

### 7.1 Graph IR

Represents the imported model at an operator graph level.

Example node types:

* `Input`
* `Linear`
* `ReLU`
* `Add`
* `LayerNorm`
* `Output`

Properties:

* tensor shapes,
* dtypes,
* parameter references,
* graph edges.

Purpose:

* correctness-oriented model representation,
* input to compiler passes.

### 7.2 Spatial IR

Represents the graph after spatial partitioning.

Adds:

* region assignments,
* operator tiling decisions,
* weight placement,
* tensor partitioning strategy.

Example questions answered here:

* which mesh region owns `Linear_1`?
* how is the hidden dimension sharded?
* where do weight tiles live?
* where are reductions performed?

### 7.3 Route IR

Represents connectivity between regions and PEs.

Contains:

* logical routes,
* source and sink endpoints,
* hop paths or route descriptors,
* message classes,
* buffering assumptions.

### 7.4 Schedule IR

Represents execution order and activation movement semantics.

Contains:

* readiness conditions,
* task dependency edges,
* pipelining constraints,
* optional timing estimates.

### 7.5 Runtime Program

The final low-level form consumed by Rust.

Contains:

* PE-local task lists,
* local memory allocations,
* routing entries,
* weight buffers,
* static operator descriptors.

---

## 8. Compiler design

### 8.1 Compiler pipeline

The compiler should initially have the following stages:

1. **Import**

   * Load a small model description from Python.
   * Produce Graph IR.

2. **Normalization**

   * Canonicalize graph forms.
   * Fuse trivial ops where useful.

3. **Partitioning / placement**

   * Assign operators or operator fragments to mesh regions.

4. **Weight layout**

   * Split and place weights into PE-local SRAM.

5. **Routing**

   * Generate routes for inter-region activation movement.

6. **Scheduling**

   * Define readiness and forwarding behavior.

7. **Artifact emission**

   * Serialize the runtime program.

### 8.2 Placement strategy for v1

Use a simple heuristic:

* one operator per contiguous rectangular region,
* region sizes are determined by output dimension and local memory needs,
* layers are laid out left-to-right across the mesh when possible,
* reductions happen in dedicated edge or collector PEs.

This should be intentionally simple. We can improve placement later.

### 8.3 Weight layout strategy for v1

For a `Linear(in_dim, out_dim)` operator:

* partition weights by output rows across region PEs,
* each PE holds a tile of the row space,
* input activation fragments are broadcast or routed to the PEs holding the relevant rows,
* partial outputs are either emitted directly or reduced locally depending on the partition scheme.

### 8.4 Routing strategy for v1

Use deterministic shortest-path routing on a 2D Manhattan mesh.

Start with:

* single message class,
* single routing policy,
* no adaptive routing,
* bounded queue depth,
* explicit hop counting for profiling.

### 8.5 Scheduling strategy for v1

Use message-driven readiness:

* a task fires when all required inputs have arrived,
* tasks consume input payloads,
* tasks update local state,
* tasks optionally emit messages to downstream PEs.

This gives a clean event-driven simulator model without cycle accuracy.

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

### 11.1 Initial API scope

The first API should be intentionally small.

Endpoints:

* `POST /v1/infer`
* `GET /healthz`
* `GET /v1/models`

Optional later endpoint:

* `POST /v1/completions`

The first version can simply accept numeric tensors rather than text prompts.

### 11.2 Suggested payload

```json
{
  "model": "mlp_tiny",
  "inputs": [0.1, 0.2, 0.3, 0.4],
  "trace": false
}
```

### 11.3 API implementation

Use **FastAPI** in Python.

Rationale:

* quick iteration,
* clean request/response models,
* good fit with uv,
* good enough for local development.

---

## 12. Observability and debugging

This project needs strong introspection or it will become opaque.

### 12.1 Required diagnostics in v1

* mesh layout dump,
* region assignment dump,
* route dump,
* local SRAM usage by PE,
* total message count,
* per-route hop counts,
* per-operator latency,
* total inference latency,
* output correctness comparison vs reference.

### 12.2 Nice-to-have diagnostics

* timeline view of events,
* heatmap of PE utilization,
* queue occupancy histogram,
* routing contention visualization,
* step-by-step trace replay.

### 12.3 Debug philosophy

Prefer plain data dumps first.

Do not start with an elaborate UI. A few good machine-readable trace files plus small Python plotting helpers are enough.

---

## 13. Project structure

Initial project structure:

```text
meshflow/
├── pyproject.toml
├── uv.lock
├── README.md
├── .python-version
├── rust-toolchain.toml
├── Cargo.toml
├── Cargo.lock
├── crates/
│   └── mesh_runtime/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── mesh.rs
│           ├── pe.rs
│           ├── message.rs
│           ├── route.rs
│           ├── task.rs
│           ├── event.rs
│           ├── runtime.rs
│           ├── artifact.rs
│           └── profiling.rs
├── python/
│   └── meshflow/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── server.py
│       ├── compiler/
│       │   ├── __init__.py
│       │   ├── import_model.py
│       │   ├── graph_ir.py
│       │   ├── spatial_ir.py
│       │   ├── route_ir.py
│       │   ├── schedule_ir.py
│       │   ├── placement.py
│       │   ├── routing.py
│       │   ├── scheduling.py
│       │   ├── lowering.py
│       │   └── artifact.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── mlp.py
│       │   └── reference.py
│       ├── runtime/
│       │   ├── __init__.py
│       │   └── runner.py
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── compile_model.py
│       │   ├── run_infer.py
│       │   └── dump_trace.py
│       ├── viz/
│       │   ├── __init__.py
│       │   ├── heatmap.py
│       │   └── timeline.py
│       └── utils/
│           ├── __init__.py
│           └── serialization.py
├── tests/
│   ├── python/
│   │   ├── test_graph_ir.py
│   │   ├── test_placement.py
│   │   ├── test_routing.py
│   │   ├── test_artifact_roundtrip.py
│   │   ├── test_reference_mlp.py
│   │   └── test_end_to_end_mlp.py
│   └── rust/
│       └── integration.rs
├── artifacts/
│   ├── compiled/
│   └── traces/
└── scripts/
    ├── dev.sh
    ├── test.sh
    └── lint.sh
```

Notes:

* `python/` contains the Python package.
* `crates/mesh_runtime/` contains the Rust simulator core.
* `artifacts/` stores generated compiled programs and traces.
* `tests/python/` and `tests/rust/` keep test boundaries explicit.

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

### 14.4 Recommended Python dependencies

Core:

* `fastapi`
* `uvicorn`
* `pydantic`
* `numpy`
* `orjson`

Compiler / tooling:

* `networkx` (optional, useful early for graph work)
* `rich` (optional for dumps)
* `typer` (optional for CLIs)

Testing:

* `pytest`
* `pytest-cov`

Linting / formatting:

* `ruff`
* `mypy`

Build:

* `maturin`

Optional later:

* `matplotlib` for simple visualization
* `torch` only if we want PyTorch import and numerical reference support

### 14.5 Recommended Rust dependencies

Initial candidates:

* `pyo3`
* `serde`
* `serde_json`
* `thiserror`
* `anyhow`
* `smallvec`
* `hashbrown`
* `petgraph` only if it actually earns its keep

Keep the Rust dependency set conservative.

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

### 15.3 Running compiler tooling locally

```bash
uv run python -m meshflow.tools.compile_model
uv run python -m meshflow.tools.run_infer
```

### 15.4 Rebuilding Rust after changes

```bash
uv run maturin develop
```

Later we can add a watcher, but this is enough initially.

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

### 16.4 Golden tests

We should add golden tests for compiled artifacts.

Examples:

* compile a tiny 2-layer MLP,
* snapshot the emitted region layout and routes,
* fail if a compiler change unexpectedly alters the artifact.

### 16.5 Numerical correctness tests

For every supported operator and model:

* run a reference Python implementation,
* run the compiled simulator path,
* compare outputs within a small tolerance.

These tests are mandatory.

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

## Milestone 0: repository bootstrap

### Objective

Create the repo skeleton, dependency setup, and dev workflow.

### Deliverables

* `pyproject.toml` with uv-managed dependencies,
* Cargo workspace and `mesh_runtime` crate,
* PyO3 + maturin wired up,
* basic package import works,
* CI-ready test commands documented,
* README with setup instructions.

### Exit criteria

The following commands work:

```bash
uv sync
uv run maturin develop
uv run python -c "import meshflow"
cargo test -p mesh_runtime
uv run pytest tests/python
```

---

## Milestone 1: minimal mesh simulator

### Objective

Build the Rust runtime skeleton and a minimal event-driven mesh.

### Deliverables

* PE and mesh data structures,
* message queues,
* deterministic routing on a 2D grid,
* event loop,
* a trivial task type such as `ForwardActivation`,
* profiling counters for hops and events.

### Exit criteria

* a message can enter one PE, traverse a route, and exit another PE,
* route latency and hop counts are reported,
* Rust unit tests cover the basic execution path.

---

## Milestone 2: Graph IR and compiler skeleton

### Objective

Build the Python-side compiler pipeline for a tiny MLP.

### Deliverables

* Graph IR definitions,
* simple model import for a hand-authored MLP,
* Spatial IR,
* naive rectangular placement,
* Route IR generation,
* artifact serialization.

### Exit criteria

* a tiny MLP can be compiled into a serialized artifact,
* layout, routing, and weight placement dumps are inspectable,
* artifact round-trip tests pass.

---

## Milestone 3: first real operator execution

### Objective

Execute a real linear layer on the mesh.

### Deliverables

* `Linear` operator task implementation in Rust,
* PE-local weight storage,
* input activation delivery,
* output fragment generation,
* Python reference implementation for comparison.

### Exit criteria

* simulator output for one linear layer matches Python reference within tolerance,
* profiling reports at least operator latency and message counts,
* end-to-end test covers compile -> load -> run -> compare.

---

## Milestone 4: multi-layer MLP pipeline

### Objective

Support a sequence of linear and activation operators across multiple mesh regions.

### Deliverables

* `ReLU` task implementation,
* multi-region execution,
* inter-layer activation routing,
* end-to-end compiled MLP execution,
* correctness and trace tests.

### Exit criteria

* a 2- or 3-layer MLP runs end to end,
* outputs match the reference model,
* activation flow across the mesh is visible in dumps or trace files.

---

## Milestone 5: inference API

### Objective

Expose the engine through a small HTTP API.

### Deliverables

* FastAPI server,
* model loading,
* `POST /v1/infer`,
* optional trace flag,
* basic health endpoint,
* local manual integration test.

### Exit criteria

* a client can send an input tensor and receive output from the simulator,
* trace mode returns useful execution metadata,
* API tests pass.

---

## Milestone 6: observability and profiling

### Objective

Make the system explain itself.

### Deliverables

* per-PE memory usage dump,
* route heatmap data,
* per-operator latency summary,
* event timeline export,
* Python helper scripts to visualize results.

### Exit criteria

* a run produces enough artifacts to explain performance bottlenecks,
* simple visualizations can be generated locally,
* debugging a bad route or placement decision is practical.

---

## Milestone 7: richer operator set or tiny decoder block

### Objective

Extend beyond the MLP once the core architecture is stable.

### Deliverables

One of the following paths:

1. add residual/add/norm support, or
2. implement a tiny decoder block with simplified attention.

### Exit criteria

* the system supports at least one architecture more interesting than an MLP,
* the compiler remains comprehensible,
* tests continue to cover numerical correctness.

---

## 19. Suggested implementation order inside milestones

Within the milestones above, the likely coding order should be:

1. repo bootstrap,
2. Rust mesh and route primitives,
3. Python IR definitions,
4. artifact format,
5. one simple message-forwarding end-to-end path,
6. linear operator execution,
7. multi-layer MLP,
8. API layer,
9. traces and visualization,
10. richer ops.

This is important. The first working vertical slice should be extremely small.

---

## 20. Risks

### 20.1 Biggest technical risk

The main risk is trying to make the simulator too realistic too early.

Mitigation:

* keep the timing model coarse,
* keep routing deterministic,
* keep the operator set tiny.

### 20.2 Biggest project risk

The main project risk is letting the compiler and runtime co-evolve without a stable artifact boundary.

Mitigation:

* define the compiled artifact format early,
* add round-trip and golden tests,
* version the artifact format if needed.

### 20.3 Another risk

The Python/Rust boundary could become chatty and awkward.

Mitigation:

* pass whole compiled artifacts, not tiny callbacks,
* keep the simulator loop entirely in Rust,
* return summary data rather than per-event Python objects by default.

---

## 21. Open questions

These do not block Milestone 0 or 1.

* Should batch size be fixed to 1 initially, or should we support a tiny microbatch abstraction from the start?
* Should routes be stored explicitly as hop lists, or generated from source/sink coordinates at load time?
* Should the artifact format be human-readable JSON first, or a binary format from the beginning?
* How much PyTorch support do we want in the importer, versus hand-authored model definitions?
* Should we represent tensors densely everywhere, or introduce fragment views early?

Recommended answers for now:

* batch size 1,
* explicit routes for easier debugging,
* JSON + binary blobs,
* hand-authored models first,
* dense fragments first.

---

## 22. Initial `pyproject.toml` shape

This is illustrative rather than final.

```toml
[project]
name = "meshflow"
version = "0.1.0"
description = "Cerebras-inspired spatial inference engine"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "pydantic>=2.8",
  "numpy>=2.0",
  "orjson>=3.10",
]

[dependency-groups]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "ruff>=0.6",
  "mypy>=1.11",
  "maturin>=1.7",
]

[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[tool.maturin]
module-name = "meshflow._mesh_runtime"
python-source = "python"
manifest-path = "crates/mesh_runtime/Cargo.toml"

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests/python"]
```

We can refine this once the actual packaging path is wired up.

---

## 23. Initial scripts

### `scripts/test.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cargo test -p mesh_runtime
uv run pytest tests/python
```

### `scripts/lint.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
uv run ruff check python tests
uv run ruff format --check python tests
uv run mypy python/meshflow
cargo fmt --check
cargo clippy -p mesh_runtime -- -D warnings
```

### `scripts/dev.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
uv sync
uv run maturin develop
uv run uvicorn meshflow.api.server:app --reload
```

---

## 24. Definition of done for v1

Meshflow v1 is done when all of the following are true:

* a small MLP can be compiled into a spatial artifact,
* the Rust simulator loads the artifact and executes it,
* weights are resident in PE-local memory,
* activations traverse explicit routes across the mesh,
* outputs match a Python reference implementation,
* the system is callable through an HTTP API,
* traces and profiling are sufficient to explain the run,
* the repo has repeatable setup, lint, and test commands.

---

## 25. Next step

The immediate next step after approving this design doc is:

**Implement Milestone 0 and Milestone 1 only.**

Do not start by importing real models.
Do not start by adding attention.
Do not start by polishing the API.

The first concrete target is a tiny Rust mesh runtime that can route and deliver activation messages, plus a Python package skeleton that can compile a trivial graph into a loadable artifact.
