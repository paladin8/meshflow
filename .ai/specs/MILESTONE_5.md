# Milestone 5: Inference API Layer

## Objective

Expose the compiler and Rust runtime through an HTTP API so that clients can compile graphs into artifacts, store them server-side, and run inference against stored artifacts — all via JSON requests. The API is a thin FastAPI layer; no database, no background workers.

## Exit criteria

1. `POST /compile` accepts a graph + weights + optional config, compiles, stores the artifact on disk, and returns an artifact ID with metadata.
2. `POST /run/{artifact_id}` loads a stored artifact, executes it with provided inputs, and returns outputs + profiling.
3. `GET /artifacts` lists stored artifact IDs.
4. `DELETE /artifacts/{artifact_id}` removes a stored artifact.
5. `GET /health` returns server status and version.
6. Errors return appropriate HTTP status codes (400, 404, 422, 500).
7. All existing compiler and runtime tests continue to pass.
8. API integration tests cover the happy path and error cases.
9. All linters clean (`cargo fmt`, `clippy`, `ruff`, `mypy`).

---

## 1. Architecture

The API is a synchronous FastAPI application wrapping the existing `compile()` and `run_program()` functions. Artifacts are stored as `.msgpack` files on disk, keyed by UUID. No database, no session state beyond the filesystem.

### 1.1 File layout

| File | Responsibility |
|------|---------------|
| `python/meshflow/api/server.py` | FastAPI app, endpoint handlers, error handling |
| `python/meshflow/api/schemas.py` | Pydantic request/response models |
| `python/meshflow/api/store.py` | Artifact file storage (save/load/delete/list) |
| `tests/python/api/test_store.py` | Unit tests for artifact store |
| `tests/python/api/test_api.py` | Integration tests using FastAPI TestClient |

### 1.2 Dependencies

FastAPI and Uvicorn are already in `pyproject.toml`. Add `httpx` as a dev dependency (required by FastAPI's `TestClient`).

---

## 2. Endpoints

### 2.1 `POST /compile`

Accepts a graph definition, weights, and optional compiler config. Compiles through all passes, stores the artifact, and returns metadata.

**Request body:**

```json
{
  "graph": {
    "nodes": [
      {"id": "l1", "op": "linear", "attrs": {"in_features": 3, "out_features": 4}},
      {"id": "r1", "op": "relu"},
      {"id": "l2", "op": "linear", "attrs": {"in_features": 4, "out_features": 2}}
    ],
    "edges": [
      {"src_node": "l1", "src_slot": 0, "dst_node": "r1", "dst_slot": 0},
      {"src_node": "r1", "src_slot": 0, "dst_node": "l2", "dst_slot": 0}
    ]
  },
  "weights": {
    "l1": {"weight": [[0.1, 0.2, 0.3], ...], "bias": [0.01, ...]},
    "l2": {"weight": [[0.4, 0.5, 0.6, 0.7], ...], "bias": [0.02, ...]}
  },
  "config": {
    "placement": "sequential",
    "routing": "xy",
    "mesh_height": 4
  }
}
```

All fields in `config` are optional. `weights` is required when the graph contains LINEAR nodes.

**String conventions:**
- `op` values use the `OpType` enum's lowercase string values: `"forward"`, `"collect"`, `"linear"`, `"relu"`.
- `config.placement` uses `PlacementStrategy` enum values: `"sequential"`.
- `config.routing` uses `RoutingStrategy` enum values: `"xy"`.

The handler maps strings to enums via `OpType(value)`, `PlacementStrategy(value)`, and `RoutingStrategy(value)`. Invalid values produce a 400 error.

**Response `201`:**

```json
{
  "artifact_id": "a1b2c3d4-...",
  "mesh_width": 2,
  "mesh_height": 4,
  "num_pes": 7,
  "input_names": ["l1"]
}
```

### 2.2 `POST /run/{artifact_id}`

Loads a stored artifact, injects inputs, runs the Rust simulator, and returns outputs with profiling.

**Request body:**

```json
{
  "inputs": {
    "l1": [1.0, 2.0, 3.0]
  }
}
```

**Response `200`:**

```json
{
  "outputs": {
    "2,4": [0.42, -0.17]
  },
  "profile": {
    "total_hops": 24,
    "total_messages": 12,
    "total_events_processed": 48,
    "total_tasks_executed": 9,
    "final_timestamp": 15
  }
}
```

Output keys are stringified coordinates (`"x,y"`). All profiling fields are always included.

### 2.3 `GET /artifacts`

**Response `200`:**

```json
{
  "artifact_ids": ["a1b2c3d4-...", "e5f6g7h8-..."]
}
```

### 2.4 `DELETE /artifacts/{artifact_id}`

**Response `200`:**

```json
{
  "deleted": "a1b2c3d4-..."
}
```

Returns `404` if the artifact does not exist.

### 2.5 `GET /health`

**Response `200`:**

```json
{
  "status": "ok",
  "runtime_version": "0.1.0"
}
```

Uses the existing `runtime_version()` function from the Rust bridge.

---

## 3. Data flow

### 3.1 Compile path

1. Pydantic `CompileRequest` validates the incoming JSON.
2. Handler converts Pydantic models to compiler types:
   - `op` strings → `OpType(value)` (e.g., `"linear"` → `OpType.LINEAR`).
   - Weight nested lists → `np.array(value, dtype=np.float64)` for each `"weight"` and `"bias"` entry.
   - Config strings → enum values via `PlacementStrategy(value)`, `RoutingStrategy(value)`.
3. Calls `compile(graph, config, weights)` → `RuntimeProgram`.
4. Calls `serialize(program)` → bytes.
5. Writes bytes to `artifacts/compiled/{uuid}.msgpack`.
6. Extracts metadata from `RuntimeProgram`:
   - `mesh_width` = `program.mesh_config.width`
   - `mesh_height` = `program.mesh_config.height`
   - `num_pes` = `len(program.pe_programs)`
   - `input_names` = `[s.name for s in program.input_slots]`
7. Returns `CompileResponse`.

### 3.2 Run path

1. Pydantic `RunRequest` validates input payloads.
2. Handler loads artifact bytes from `artifacts/compiled/{artifact_id}.msgpack`.
3. Calls `run_program(artifact_bytes, inputs)` → `SimResult`.
4. Converts `SimResult` outputs (coord tuples → `"x,y"` strings) and profiling fields.
5. Returns `RunResponse`.

### 3.3 Conversion layer

Pydantic-to-compiler-type conversion lives inline in the endpoint handlers. It maps JSON field names to dataclass constructors — straightforward enough not to need its own module.

---

## 4. Artifact store

`ArtifactStore` is a class that manages `.msgpack` files in a base directory.

### 4.1 Interface

```python
class ArtifactStore:
    def __init__(self, base_dir: Path): ...
    def save(self, artifact_id: str, data: bytes) -> Path: ...
    def load(self, artifact_id: str) -> bytes: ...
    def delete(self, artifact_id: str) -> None: ...
    def list(self) -> list[str]: ...
    def exists(self, artifact_id: str) -> bool: ...
```

- `save` writes to `{base_dir}/{artifact_id}.msgpack`, creating the directory if needed.
- `load` reads and returns the bytes. Raises `FileNotFoundError` if the file does not exist.
- `delete` removes the file. Raises `FileNotFoundError` if it does not exist.
- `list` returns artifact IDs (filenames without `.msgpack` extension), sorted alphabetically.

### 4.2 Server integration

The FastAPI app creates one `ArtifactStore` instance at startup. The base directory defaults to `artifacts/compiled/` relative to the project root, configurable via the `MESHFLOW_ARTIFACT_DIR` environment variable.

The store is exposed via a FastAPI dependency (`Depends(get_store)`) so that tests can override it with a `tmp_path`-based store using `app.dependency_overrides`.

---

## 5. Error handling

| Condition | HTTP status | Response body |
|-----------|-------------|--------------|
| Invalid JSON / missing required fields | 422 | FastAPI default validation error |
| `GraphIR.validate()` raises `ValueError` | 400 | `{"detail": "..."}` |
| `_validate_weights()` raises `ValueError` | 400 | `{"detail": "..."}` |
| Shape mismatch in `_validate_shape_chaining()` | 400 | `{"detail": "..."}` |
| Unknown artifact ID in `/run` or `/delete` | 404 | `{"detail": "artifact not found: ..."}` |
| Unknown input name in `/run` | 400 | `{"detail": "..."}` |
| Rust runtime error | 500 | `{"detail": "runtime error: ..."}` |

**Exception mapping:**
- `ValueError` from the compiler (`validate()`, `_validate_weights()`, `_validate_shape_chaining()`) → 400.
- `FileNotFoundError` from the artifact store → 404.
- `RuntimeError` from `run_program()` (PyO3 maps Rust errors to `RuntimeError`) → 500. This includes unknown input names, which the Rust runtime validates via `UnknownInputSlot` — surfaced as a 400 by checking the error message for "unknown input slot".
- All exception handlers registered via `@app.exception_handler` or try/except in the endpoint.

---

## 6. Pydantic schemas

### 6.1 Request models

```python
class NodeInput(BaseModel):
    id: str
    op: str
    attrs: dict[str, Any] | None = None

class EdgeInput(BaseModel):
    src_node: str
    src_slot: int
    dst_node: str
    dst_slot: int

class GraphInput(BaseModel):
    nodes: list[NodeInput]
    edges: list[EdgeInput]

class ConfigInput(BaseModel):
    placement: str | None = None
    routing: str | None = None
    mesh_height: int | None = None
    mesh_width: int | None = None

class CompileRequest(BaseModel):
    graph: GraphInput
    weights: dict[str, dict[str, list[Any]]] | None = None
    config: ConfigInput | None = None

class RunRequest(BaseModel):
    inputs: dict[str, list[float]]
```

### 6.2 Response models

```python
class CompileResponse(BaseModel):
    artifact_id: str
    mesh_width: int
    mesh_height: int
    num_pes: int
    input_names: list[str]

class ProfileResponse(BaseModel):
    total_hops: int
    total_messages: int
    total_events_processed: int
    total_tasks_executed: int
    final_timestamp: int

class RunResponse(BaseModel):
    outputs: dict[str, list[float]]
    profile: ProfileResponse

class ArtifactListResponse(BaseModel):
    artifact_ids: list[str]

class DeleteResponse(BaseModel):
    deleted: str

class HealthResponse(BaseModel):
    status: str
    runtime_version: str
```

---

## 7. Testing

### 7.1 Store unit tests (`tests/python/api/test_store.py`)

- Save and load round-trip: bytes in == bytes out.
- List returns stored IDs.
- Delete removes the file; subsequent load raises `FileNotFoundError`.
- Load missing artifact raises `FileNotFoundError`.
- Exists returns True/False correctly.
- Uses `tmp_path` fixture — no real `artifacts/` directory touched.

### 7.2 API integration tests (`tests/python/api/test_api.py`)

Using FastAPI `TestClient` with a `tmp_path` store (override the store dependency to use a temp directory):

- **Compile simple graph:** `POST /compile` with FORWARD→COLLECT (no weights). Assert 201, artifact ID present, correct mesh dimensions.
- **Compile MLP:** `POST /compile` with LINEAR→RELU→LINEAR + weights. Assert 201, metadata correct (mesh width, height, input names).
- **Compile invalid graph:** Missing attrs on LINEAR. Assert 400.
- **Run valid artifact:** Compile, then `POST /run/{id}` with inputs. Assert 200, outputs present, all profile fields present.
- **Run missing artifact:** `POST /run/nonexistent`. Assert 404.
- **Run wrong input name:** Compile, then run with wrong input key. Assert 400.
- **List artifacts:** Compile two graphs, `GET /artifacts`. Assert both IDs present.
- **Delete artifact:** Compile, delete, subsequent run returns 404.
- **Health check:** `GET /health`. Assert 200, status "ok", version present.

### 7.3 What the API tests do NOT cover

Numerical correctness — that's already verified by `tests/python/runtime/test_end_to_end.py`. The API tests verify HTTP semantics, serialization, and error handling.

---

## 8. Scope boundaries

**In scope:**

- Five endpoints: compile, run, list, delete, health.
- Pydantic request/response validation.
- File-based artifact storage.
- Inline weight submission as JSON arrays.
- Profiling always included in run response.
- Error handling with appropriate HTTP status codes.
- Integration tests using TestClient.

**Out of scope (deferred):**

- Per-PE stats in run response (`SimResult.pe_stats` is available but omitted for now; aggregate counters are sufficient for M5. Per-PE stats are a natural addition in the observability milestone).
- Authentication / authorization.
- Artifact TTL or automatic cleanup.
- Multipart weight upload.
- Combined compile-and-run `/infer` endpoint.
- WebSocket streaming of profiling data.
- Rate limiting.
- Persistent artifact metadata (names, timestamps, descriptions).
- API versioning (v1 prefix).
