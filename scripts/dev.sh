#!/usr/bin/env bash
set -euo pipefail
uv sync
uv run maturin develop
uv run uvicorn meshflow.api.server:app --reload
