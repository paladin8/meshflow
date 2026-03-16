#!/usr/bin/env bash
set -euo pipefail
uv run ruff check python tests
uv run ruff format --check python tests
uv run mypy python/meshflow
cargo fmt --check
cargo clippy -p mesh_runtime -- -D warnings
