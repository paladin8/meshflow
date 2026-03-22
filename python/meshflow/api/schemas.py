"""Pydantic request/response models for the meshflow API."""

from typing import Any

from pydantic import BaseModel


# --- Request models ---


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


# --- Response models ---


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
