#!/usr/bin/env bash
set -euo pipefail
cargo test -p mesh_runtime
uv run pytest tests/python
