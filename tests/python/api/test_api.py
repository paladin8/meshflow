"""Integration tests for the meshflow API."""

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meshflow.api.server import app, get_store
from meshflow.api.store import ArtifactStore


@pytest.fixture()
def client(tmp_path: Path) -> Generator[TestClient]:
    store = ArtifactStore(tmp_path)
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def _simple_graph_body() -> dict:
    """FORWARD→COLLECT graph, no weights needed."""
    return {
        "graph": {
            "nodes": [
                {"id": "a", "op": "forward"},
                {"id": "b", "op": "collect"},
            ],
            "edges": [
                {"src_node": "a", "src_slot": 0, "dst_node": "b", "dst_slot": 0},
            ],
        }
    }


def _mlp_graph_body() -> dict:
    """LINEAR→RELU→LINEAR graph with weights."""
    return {
        "graph": {
            "nodes": [
                {"id": "l1", "op": "linear", "attrs": {"in_features": 2, "out_features": 3}},
                {"id": "r1", "op": "relu"},
                {"id": "l2", "op": "linear", "attrs": {"in_features": 3, "out_features": 2}},
            ],
            "edges": [
                {"src_node": "l1", "src_slot": 0, "dst_node": "r1", "dst_slot": 0},
                {"src_node": "r1", "src_slot": 0, "dst_node": "l2", "dst_slot": 0},
            ],
        },
        "weights": {
            "l1": {
                "weight": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                "bias": [0.01, 0.02, 0.03],
            },
            "l2": {
                "weight": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "bias": [0.01, 0.02],
            },
        },
    }


class TestCompile:
    def test_compile_simple_graph(self, client: TestClient) -> None:
        resp = client.post("/compile", json=_simple_graph_body())
        assert resp.status_code == 201
        data = resp.json()
        assert "artifact_id" in data
        assert data["mesh_width"] == 2
        assert data["mesh_height"] == 1
        assert data["num_pes"] == 2
        assert data["input_names"] == ["a"]

    def test_compile_mlp(self, client: TestClient) -> None:
        resp = client.post("/compile", json=_mlp_graph_body())
        assert resp.status_code == 201
        data = resp.json()
        assert data["mesh_width"] == 2
        assert data["mesh_height"] >= 2
        assert data["num_pes"] >= 4
        assert "l1" in data["input_names"]

    def test_compile_invalid_graph(self, client: TestClient) -> None:
        body = {
            "graph": {
                "nodes": [
                    {"id": "l1", "op": "linear"},  # missing attrs
                ],
                "edges": [],
            },
            "weights": {
                "l1": {
                    "weight": [[0.1]],
                    "bias": [0.01],
                },
            },
        }
        resp = client.post("/compile", json=body)
        assert resp.status_code == 400

    def test_compile_invalid_op(self, client: TestClient) -> None:
        body = {
            "graph": {
                "nodes": [{"id": "a", "op": "bogus"}],
                "edges": [],
            }
        }
        resp = client.post("/compile", json=body)
        assert resp.status_code == 400


class TestRun:
    def test_run_valid_artifact(self, client: TestClient) -> None:
        compile_resp = client.post("/compile", json=_simple_graph_body())
        artifact_id = compile_resp.json()["artifact_id"]

        run_resp = client.post(f"/run/{artifact_id}", json={"inputs": {"a": [1.0, 2.0]}})
        assert run_resp.status_code == 200
        data = run_resp.json()
        assert "outputs" in data
        # FORWARD→COLLECT passes input through unchanged
        assert data["outputs"]["1,0"] == [1.0, 2.0]
        assert "profile" in data
        profile = data["profile"]
        assert "total_hops" in profile
        assert "total_messages" in profile
        assert "total_events_processed" in profile
        assert "total_tasks_executed" in profile
        assert "final_timestamp" in profile

    def test_run_missing_artifact(self, client: TestClient) -> None:
        resp = client.post("/run/nonexistent", json={"inputs": {"a": [1.0]}})
        assert resp.status_code == 404

    def test_run_wrong_input_name(self, client: TestClient) -> None:
        compile_resp = client.post("/compile", json=_simple_graph_body())
        artifact_id = compile_resp.json()["artifact_id"]

        run_resp = client.post(f"/run/{artifact_id}", json={"inputs": {"wrong": [1.0]}})
        assert run_resp.status_code == 400


class TestArtifacts:
    def test_list_artifacts(self, client: TestClient) -> None:
        client.post("/compile", json=_simple_graph_body())
        client.post("/compile", json=_simple_graph_body())

        resp = client.get("/artifacts")
        assert resp.status_code == 200
        assert len(resp.json()["artifact_ids"]) == 2

    def test_delete_artifact(self, client: TestClient) -> None:
        compile_resp = client.post("/compile", json=_simple_graph_body())
        artifact_id = compile_resp.json()["artifact_id"]

        del_resp = client.delete(f"/artifacts/{artifact_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] == artifact_id

        run_resp = client.post(f"/run/{artifact_id}", json={"inputs": {"a": [1.0]}})
        assert run_resp.status_code == 404

    def test_delete_missing_artifact(self, client: TestClient) -> None:
        resp = client.delete("/artifacts/nonexistent")
        assert resp.status_code == 404


class TestHealth:
    def test_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "runtime_version" in data
