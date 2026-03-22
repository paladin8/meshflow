"""FastAPI server — compile graphs and run inference against stored artifacts."""

import os
import uuid
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException

from meshflow._mesh_runtime import run_program, runtime_version
from meshflow.compiler import CompilerConfig, Edge, GraphIR, Node, OpType, compile
from meshflow.compiler.artifact import serialize
from meshflow.compiler.config import PlacementStrategy, RoutingStrategy

from .schemas import (
    ArtifactListResponse,
    CompileRequest,
    CompileResponse,
    DeleteResponse,
    HealthResponse,
    ProfileResponse,
    RunRequest,
    RunResponse,
)
from .store import ArtifactStore

app = FastAPI(title="meshflow")

# server.py is at python/meshflow/api/server.py — four levels below repo root.
_DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "compiled"
)
_ARTIFACT_DIR = Path(os.environ.get("MESHFLOW_ARTIFACT_DIR", str(_DEFAULT_ARTIFACT_DIR)))
_store = ArtifactStore(_ARTIFACT_DIR)


def get_store() -> ArtifactStore:
    return _store


# --- Endpoints ---


@app.post("/compile", response_model=CompileResponse, status_code=201)
def compile_graph(
    req: CompileRequest, store: ArtifactStore = Depends(get_store)
) -> CompileResponse:
    try:
        graph = _build_graph(req)
        weights = _build_weights(req) if req.weights else None
        config = _build_config(req)

        program = compile(graph, config, weights)
        artifact_bytes = serialize(program)

        artifact_id = str(uuid.uuid4())
        store.save(artifact_id, artifact_bytes)

        return CompileResponse(
            artifact_id=artifact_id,
            mesh_width=program.mesh_config.width,
            mesh_height=program.mesh_config.height,
            num_pes=len(program.pe_programs),
            input_names=[s.name for s in program.input_slots],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/run/{artifact_id}", response_model=RunResponse)
def run_artifact(
    artifact_id: str, req: RunRequest, store: ArtifactStore = Depends(get_store)
) -> RunResponse:
    try:
        artifact_bytes = store.load(artifact_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    try:
        result = run_program(artifact_bytes, req.inputs)
    except RuntimeError as e:
        msg = str(e)
        if "unknown input slot" in msg.lower():
            raise HTTPException(status_code=400, detail=msg) from e
        raise HTTPException(status_code=500, detail=f"runtime error: {msg}") from e

    outputs: dict[str, list[float]] = {}
    for coord, payload in result.outputs.items():
        key = f"{coord[0]},{coord[1]}"
        outputs[key] = list(payload)

    return RunResponse(
        outputs=outputs,
        profile=ProfileResponse(
            total_hops=result.total_hops,
            total_messages=result.total_messages,
            total_events_processed=result.total_events_processed,
            total_tasks_executed=result.total_tasks_executed,
            final_timestamp=result.final_timestamp,
        ),
    )


@app.get("/artifacts", response_model=ArtifactListResponse)
def list_artifacts(store: ArtifactStore = Depends(get_store)) -> ArtifactListResponse:
    return ArtifactListResponse(artifact_ids=store.list())


@app.delete("/artifacts/{artifact_id}", response_model=DeleteResponse)
def delete_artifact(artifact_id: str, store: ArtifactStore = Depends(get_store)) -> DeleteResponse:
    try:
        store.delete(artifact_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return DeleteResponse(deleted=artifact_id)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", runtime_version=runtime_version())


# --- Conversion helpers ---


def _build_graph(req: CompileRequest) -> GraphIR:
    nodes = [
        Node(
            id=n.id,
            op=OpType(n.op),
            attrs=n.attrs,
        )
        for n in req.graph.nodes
    ]
    edges = [
        Edge(
            src_node=e.src_node,
            src_slot=e.src_slot,
            dst_node=e.dst_node,
            dst_slot=e.dst_slot,
        )
        for e in req.graph.edges
    ]
    return GraphIR(nodes=nodes, edges=edges)


def _build_weights(req: CompileRequest) -> dict[str, dict[str, np.ndarray]] | None:
    if req.weights is None:
        return None
    result: dict[str, dict[str, np.ndarray]] = {}
    for node_id, tensors in req.weights.items():
        result[node_id] = {name: np.array(data, dtype=np.float64) for name, data in tensors.items()}
    return result


def _build_config(req: CompileRequest) -> CompilerConfig:
    config = CompilerConfig()
    if req.config is not None:
        if req.config.placement is not None:
            config.placement = PlacementStrategy(req.config.placement)
        if req.config.routing is not None:
            config.routing = RoutingStrategy(req.config.routing)
        if req.config.mesh_height is not None:
            config.mesh_height = req.config.mesh_height
        if req.config.mesh_width is not None:
            config.mesh_width = req.config.mesh_width
    return config
