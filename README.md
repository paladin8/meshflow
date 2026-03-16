# Meshflow

Cerebras-inspired spatial inference engine.

Meshflow compiles small neural networks into spatial execution plans and runs them on a simulated 2D mesh of processing elements (PEs). Weights are resident in PE-local memory, activations flow along explicit routes.

## Prerequisites

- Python 3.12+
- Rust (stable toolchain)
- [uv](https://docs.astral.sh/uv/) for Python dependency management

## Setup

```bash
uv sync
uv run maturin develop
```

## Verify installation

```bash
uv run python -c "import meshflow; print(meshflow.__version__)"
```

## Running tests

```bash
# All tests
./scripts/test.sh

# Python tests only
uv run pytest tests/python

# Rust tests only
cargo test -p mesh_runtime
```

## Lint and format

```bash
./scripts/lint.sh
```

## Development server

```bash
uv run uvicorn meshflow.api.server:app --reload
```

## Rebuild Rust extension

After changing Rust code:

```bash
uv run maturin develop
```

## Project structure

```
meshflow/
├── python/meshflow/      # Python package (compiler, API, tools)
├── crates/mesh_runtime/  # Rust runtime (mesh simulator core)
├── tests/python/         # Python tests
├── tests/rust/           # Rust integration tests
├── artifacts/            # Generated compiled programs and traces
└── scripts/              # Dev workflow scripts
```
