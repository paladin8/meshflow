"""Tests for artifact serialization and inspect tool."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest
from meshflow.compiler import compile
from meshflow.compiler.artifact import (
    MeshProgramConfig,
    PEProgram,
    RuntimeProgram,
    deserialize,
    serialize,
)
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType


def _make_chain_program() -> RuntimeProgram:
    """Build a 3-node forward chain program for testing."""
    graph = GraphIR(
        nodes=[
            Node(id="a", op=OpType.FORWARD),
            Node(id="b", op=OpType.FORWARD),
            Node(id="c", op=OpType.COLLECT),
        ],
        edges=[
            Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0),
            Edge(src_node="b", src_slot=0, dst_node="c", dst_slot=0),
        ],
    )
    return compile(graph)


class TestSerializationRoundTrip:
    def test_round_trip_chain(self) -> None:
        original = _make_chain_program()
        data = serialize(original)
        restored = deserialize(data)

        assert restored.version == original.version
        assert restored.mesh_config == original.mesh_config
        assert len(restored.pe_programs) == len(original.pe_programs)
        assert len(restored.input_slots) == len(original.input_slots)

        for orig_pe, rest_pe in zip(original.pe_programs, restored.pe_programs):
            assert rest_pe.coord == orig_pe.coord
            assert rest_pe.initial_sram == orig_pe.initial_sram
            assert len(rest_pe.tasks) == len(orig_pe.tasks)
            for orig_task, rest_task in zip(orig_pe.tasks, rest_pe.tasks):
                assert rest_task.kind == orig_task.kind
                assert rest_task.trigger_slot == orig_task.trigger_slot
                assert rest_task.input_slot == orig_task.input_slot
                assert rest_task.route_dest == orig_task.route_dest
                assert rest_task.route_hops == orig_task.route_hops

        for orig_slot, rest_slot in zip(original.input_slots, restored.input_slots):
            assert rest_slot.name == orig_slot.name
            assert rest_slot.coord == orig_slot.coord
            assert rest_slot.payload_slot == orig_slot.payload_slot

    def test_round_trip_single_collect(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="x", op=OpType.COLLECT)],
            edges=[],
        )
        original = compile(graph)
        restored = deserialize(serialize(original))

        assert restored.version == 1
        assert restored.mesh_config.width == 1
        assert len(restored.pe_programs) == 1
        assert restored.pe_programs[0].tasks[0].kind == "collect_output"
        assert restored.pe_programs[0].tasks[0].route_hops is None

    def test_round_trip_with_initial_sram(self) -> None:
        """Verify initial_sram round-trips correctly (for future M3 use)."""
        program = RuntimeProgram(
            version=1,
            mesh_config=MeshProgramConfig(width=2, height=1),
            pe_programs=[
                PEProgram(
                    coord=(0, 0),
                    tasks=[],
                    initial_sram={0: [1.0, 2.0, 3.0], 1: [4.0, 5.0]},
                ),
                PEProgram(coord=(1, 0), tasks=[], initial_sram={}),
            ],
            input_slots=[],
        )
        restored = deserialize(serialize(program))

        assert restored.pe_programs[0].initial_sram == {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0]}
        assert restored.pe_programs[1].initial_sram == {}

    def test_serialized_bytes_are_msgpack(self) -> None:
        import msgpack

        program = _make_chain_program()
        data = serialize(program)
        # Should be valid msgpack
        raw = msgpack.unpackb(data, raw=False)
        assert isinstance(raw, dict)
        assert "version" in raw
        assert "mesh_config" in raw

    def test_round_trip_preserves_defaults(self) -> None:
        program = _make_chain_program()
        restored = deserialize(serialize(program))

        assert restored.mesh_config.hop_latency == 1
        assert restored.mesh_config.task_base_latency == 1
        assert restored.mesh_config.max_events == 100_000


class TestDeserializeErrors:
    def test_malformed_bytes(self) -> None:
        with pytest.raises(ValueError, match="failed to unpack"):
            deserialize(b"not valid msgpack \xff\xfe")

    def test_missing_fields(self) -> None:
        import msgpack

        data = msgpack.packb({"version": 1}, use_bin_type=True)
        with pytest.raises(ValueError, match="invalid artifact structure"):
            deserialize(data)


class TestInspectTool:
    def test_inspect_outputs_json(self) -> None:
        program = _make_chain_program()
        data = serialize(program)

        with tempfile.NamedTemporaryFile(suffix=".mpk", delete=False) as f:
            f.write(data)
            f.flush()
            tmp_path = f.name

        result = subprocess.run(
            ["uv", "run", "python", "-m", "meshflow.tools.inspect_artifact", tmp_path],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        parsed = json.loads(result.stdout)
        assert parsed["version"] == 1
        assert parsed["mesh_config"]["width"] == 3
        assert len(parsed["pe_programs"]) == 3
        assert len(parsed["input_slots"]) == 1

        Path(tmp_path).unlink()

    def test_inspect_missing_file(self) -> None:
        result = subprocess.run(
            ["uv", "run", "python", "-m", "meshflow.tools.inspect_artifact", "/nonexistent.mpk"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "not found" in result.stderr
