# Meshflow

Meshflow is a Cerebras-inspired spatial inference engine.

The project has two primary goals:

1. Compile a small neural network into a spatial execution plan.
2. Execute that plan on a simulated 2D mesh of processing elements (PEs) with resident weights and activati([docs.anthropic.com](https://docs.anthropic.com/en/docs/claude-code/memory))n- resident weights in PE-local memory,

* activations moving across explicit routes,
* Python compiler + orchestration,
* Rust mesh runtime,
* end-to-end correctness for a small MLP,
* strong tracing and profiling.

Do not expand scope unless explicitly asked.

Out of scope by default:

* LLVM backends,
* cycle-accurate hardware simulation,
* full transformer serving,
* distributed multi-host execution,
* premature performance optimization,
* large framework abstractions.

## Architecture rules

### Language split

Use Python for:

* model import,
* IR definitions,
* compiler passes,
* artifact generation,
* API layer,
* visualization and tooling.

Use Rust for:

* mesh data structures,
* PE state,
* message routing,
* event loop,
* task execution,
* profiling collection,
* compiled artifact loading.

Do not move the inner simulator loop into Python.

Keep the Python/Rust boundary narrow.
Pass whole artifacts and tensor payloads across the boundary, not many tiny callbacks.

### Execution model

The simulator is event-driven, not cycle-accurate.

Model these concepts explicitly:

* PE-local SRAM,
* message queues,
* routes over a 2D mesh,
* task readiness,
* coarse timing costs,
* operator regions,
* profiling counters.

Do not model low-level hardware details unless there is a clear need.

### Compiler model

Use custom IRs, not LLVM.

The expected flow is:

`Graph IR -> Spatial IR -> Route/Schedule IR -> Runtime Program`

Keep the compiler understandable.
Placement, routing, memory layout, and correctness matter more than compiler cleverness.

## Source tree expectations

Expected repo layout:

* `python/meshflow/` for Python package code,
* `crates/mesh_runtime/` for Rust runtime,
* `tests/python/` for Python tests,
* `tests/rust/` for Rust integration tests,
* `artifacts/compiled/` and `artifacts/traces/` for generated outputs.

Keep compiler code, runtime code, and tools clearly separated.

## Dependency and build workflow

Use `uv` for Python dependency management and command execution.
Use Cargo for Rust dependencies.
Use `maturin` for the Python/Rust bridge.

### Setup

```bash
uv sync
uv run maturin develop
```

### Rebuild Rust extension after Rust changes

```bash
uv run maturin develop
```

## Common commands

### Run Python tests

```bash
uv run pytest tests/python
```

### Run Rust tests

```bash
cargo test -p mesh_runtime
```

### Run all tests

```bash
cargo test -p mesh_runtime
uv run pytest tests/python
```

### Lint and format checks

```bash
uv run ruff check python tests
uv run ruff format --check python tests
uv run mypy python/meshflow
cargo fmt --check
cargo clippy -p mesh_runtime -- -D warnings
```

### Run local API

```bash
uv run uvicorn meshflow.api.server:app --reload
```

## Coding standards

### General

Prefer simple, explicit code.
Do not introduce abstractions until they earn their keep.
Optimize for readability as a systems artifact.

When changing behavior, update tests in the same change.
When changing architecture, keep the design document aligned.

### Python

* Use type hints aggressively.
* Prefer dataclasses or simple typed models over magic.
* Keep compiler passes small and composable.
* Avoid metaprogramming.

### Rust

* Prefer enums for message kinds, task kinds, and event kinds.
* Keep state transitions explicit.
* Use compact data structures where reasonable.
* Avoid premature generic abstractions.

## Testing and verification rules

Every meaningful change should be verified.

### Minimum expectations

If you change compiler logic:

* run relevant Python tests,
* verify artifact serialization / round-trip behavior,
* verify runtime compatibility if the artifact schema changed.

If you change runtime logic:

* run Rust tests,
* run at least one Python end-to-end test against the runtime.

If you change numerical execution:

* compare simulator outputs against the Python reference,
* use tolerance-based assertions,
* note any intentional numerical deviations.

### Numerical correctness

Numerical correctness is a first-class requirement.
The simulator path must be checked against a reference implementation for supported operators and models.

### Artifact changes

If you change the compiled artifact format:

* update both compiler and runtime together,
* update serialization tests,
* update any golden or snapshot-style tests.

## Workflow rules for Claude

When working in this repo:

1. Explore first.
2. Form a plan.
3. Make the smallest coherent change.
4. Run targeted verification.
5. Summarize what changed and what was verified.

Do not make broad speculative refactors.
Do not silently change repo structure unless necessary.
Do not add new top-level dependencies without a reason.

When implementing features, preserve the project direction:

* MLP pipeline first,
* richer operators later,
* tiny decoder block only after the MLP path is stable.

If asked to add something outside current scope, call out the tradeoff clearly.

## Profiling and observability expectations

Prefer adding introspection early.
Useful outputs include:

* mesh layout dumps,
* region assignments,
* routes,
* per-PE memory usage,
* message counts,
* hop counts,
* operator latency summaries,
* end-to-end latency,
* trace exports.

A change that makes the runtime harder to inspect is usually a bad change.

## Milestone priorities

Near-term priorities are:

1. repo bootstrap,
2. minimal Rust mesh runtime,
3. compiler skeleton and artifact format,
4. single linear operator execution,
5. multi-layer MLP pipeline,
6. API layer,
7. observability improvements.

Unless explicitly requested, work on the earliest unfinished milestone.

## When uncertain

If the correct design is unclear, prefer:

* simpler artifacts,
* deterministic routing,
* dense tensor fragments,
* batch size 1,
* explicit routes over implicit generation,
* correctness over speed,
* debuggability over sophistication.

## Definition of success for v1

Meshflow v1 is successful when:

* a small MLP compiles into a spatial artifact,
* the Rust runtime loads and executes it,
* weights are resident in PE-local memory,
* activations move along explicit routes,
* outputs match the Python reference,
* the system is callable through an HTTP API,
* traces and profiling explain the run,
* setup and test commands are repeatable.
