"""Tests for artifact serialization and inspect tool."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest
from meshflow.compiler import compile
from meshflow.compiler.artifact import (
    AddTask,
    BroadcastRouteTask,
    CollectOutputTask,
    ConcatCollectForwardTask,
    ConcatCollectTask,
    LinearTask,
    MatMulTask,
    MeshProgramConfig,
    PEProgram,
    RuntimeProgram,
    SoftmaxTask,
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
                assert type(rest_task) is type(orig_task)
                assert rest_task.kind == orig_task.kind
                assert rest_task.trigger_slot == orig_task.trigger_slot
                assert rest_task == orig_task

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
        task = restored.pe_programs[0].tasks[0]
        assert task.kind == "collect_output"
        assert isinstance(task, CollectOutputTask)

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

    def test_round_trip_sram_capacity(self) -> None:
        """Verify sram_capacity_bytes round-trips correctly."""
        program = RuntimeProgram(
            version=1,
            mesh_config=MeshProgramConfig(width=2, height=1),
            pe_programs=[
                PEProgram(coord=(0, 0), tasks=[], sram_capacity_bytes=65536),
                PEProgram(coord=(1, 0), tasks=[], sram_capacity_bytes=None),
            ],
            input_slots=[],
        )
        restored = deserialize(serialize(program))

        assert restored.pe_programs[0].sram_capacity_bytes == 65536
        assert restored.pe_programs[1].sram_capacity_bytes is None

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

        assert restored.mesh_config.task_base_latency == 1
        assert restored.mesh_config.max_events == 100_000

    def test_round_trip_linear_task(self) -> None:
        program = RuntimeProgram(
            version=1,
            mesh_config=MeshProgramConfig(width=2, height=1),
            pe_programs=[
                PEProgram(
                    coord=(0, 0),
                    tasks=[
                        LinearTask(
                            trigger_slot=0,
                            input_slot=0,
                            weight_slot=1,
                            bias_slot=2,
                            tile_rows=3,
                            tile_cols=4,
                            routes=[BroadcastRouteTask(dest=(1, 0), payload_slot=0)],
                            fragment_offset=0,
                        )
                    ],
                    initial_sram={1: [1.0, 2.0], 2: [3.0]},
                ),
                PEProgram(
                    coord=(1, 0),
                    tasks=[
                        ConcatCollectTask(
                            trigger_slot=0, num_fragments=2, total_rows=6, fragment_offset=0
                        ),
                    ],
                    initial_sram={},
                ),
            ],
            input_slots=[],
        )
        restored = deserialize(serialize(program))

        task0 = restored.pe_programs[0].tasks[0]
        assert isinstance(task0, LinearTask)
        assert task0.tile_rows == 3
        assert task0.tile_cols == 4
        assert task0.routes[0].dest == (1, 0)
        assert task0.routes[0].payload_slot == 0
        assert task0.fragment_offset == 0
        assert restored.pe_programs[0].initial_sram == {1: [1.0, 2.0], 2: [3.0]}

        task1 = restored.pe_programs[1].tasks[0]
        assert isinstance(task1, ConcatCollectTask)
        assert task1.num_fragments == 2
        assert task1.total_rows == 6
        assert task1.fragment_offset == 0

    def test_round_trip_concat_collect_forward_task(self) -> None:
        program = RuntimeProgram(
            version=1,
            mesh_config=MeshProgramConfig(width=2, height=4),
            pe_programs=[
                PEProgram(
                    coord=(0, 3),
                    tasks=[
                        ConcatCollectForwardTask(
                            trigger_slot=0,
                            num_fragments=3,
                            total_rows=6,
                            fragment_offset=0,
                            activation="relu",
                            routes=[
                                BroadcastRouteTask(dest=(1, 0)),
                                BroadcastRouteTask(dest=(1, 1)),
                                BroadcastRouteTask(dest=(1, 2)),
                            ],
                        ),
                    ],
                    initial_sram={},
                ),
            ],
            input_slots=[],
        )
        restored = deserialize(serialize(program))

        task = restored.pe_programs[0].tasks[0]
        assert isinstance(task, ConcatCollectForwardTask)
        assert task.kind == "concat_collect_forward"
        assert task.num_fragments == 3
        assert task.total_rows == 6
        assert task.fragment_offset == 0
        assert task.activation == "relu"
        assert len(task.routes) == 3
        assert task.routes[0].dest == (1, 0)
        assert task.routes[0].dest == (1, 0)
        assert task.routes[1].dest == (1, 1)
        assert task.routes[1].dest == (1, 1)
        assert task.routes[2].dest == (1, 2)
        assert task.routes[2].dest == (1, 2)

    def test_round_trip_concat_collect_forward_scatter(self) -> None:
        program = RuntimeProgram(
            version=1,
            mesh_config=MeshProgramConfig(width=2, height=4),
            pe_programs=[
                PEProgram(
                    coord=(0, 3),
                    tasks=[
                        ConcatCollectForwardTask(
                            trigger_slot=0,
                            num_fragments=3,
                            total_rows=6,
                            fragment_offset=0,
                            num_positions=2,
                            scatter=True,
                            activation="relu",
                            routes=[
                                BroadcastRouteTask(dest=(1, 0)),
                                BroadcastRouteTask(dest=(1, 1)),
                            ],
                        ),
                    ],
                    initial_sram={},
                ),
            ],
            input_slots=[],
        )
        restored = deserialize(serialize(program))

        task = restored.pe_programs[0].tasks[0]
        assert isinstance(task, ConcatCollectForwardTask)
        assert task.kind == "concat_collect_forward"
        assert task.num_fragments == 3
        assert task.total_rows == 6
        assert task.fragment_offset == 0
        assert task.num_positions == 2
        assert task.scatter is True
        assert task.activation == "relu"
        assert len(task.routes) == 2
        assert task.routes[0].dest == (1, 0)
        assert task.routes[0].dest == (1, 0)
        assert task.routes[1].dest == (1, 1)
        assert task.routes[1].dest == (1, 1)

    def test_round_trip_add_task(self) -> None:
        program = RuntimeProgram(
            version=1,
            mesh_config=MeshProgramConfig(width=3, height=1),
            pe_programs=[
                PEProgram(
                    coord=(1, 0),
                    tasks=[
                        AddTask(
                            trigger_slot=1,
                            input_slot_a=0,
                            input_slot_b=1,
                            output_slot=2,
                            routes=[
                                BroadcastRouteTask(dest=(2, 0), payload_slot=0),
                                BroadcastRouteTask(dest=(2, 1), payload_slot=1),
                            ],
                        ),
                    ],
                    initial_sram={},
                ),
            ],
            input_slots=[],
        )
        restored = deserialize(serialize(program))

        task = restored.pe_programs[0].tasks[0]
        assert isinstance(task, AddTask)
        assert task.kind == "add"
        assert task.trigger_slot == 1
        assert task.input_slot_a == 0
        assert task.input_slot_b == 1
        assert task.output_slot == 2
        assert len(task.routes) == 2
        assert task.routes[0].dest == (2, 0)
        assert task.routes[0].dest[0] >= 0  # valid dest
        assert task.routes[0].payload_slot == 0
        assert task.routes[1].dest == (2, 1)
        assert task.routes[1].payload_slot == 1

    def test_round_trip_softmax_task(self) -> None:
        program = RuntimeProgram(
            version=1,
            mesh_config=MeshProgramConfig(width=1, height=1),
            pe_programs=[
                PEProgram(
                    coord=(0, 0),
                    tasks=[
                        SoftmaxTask(trigger_slot=5, input_slot=5, output_slot=6),
                    ],
                    initial_sram={},
                ),
            ],
            input_slots=[],
        )
        restored = deserialize(serialize(program))

        task = restored.pe_programs[0].tasks[0]
        assert isinstance(task, SoftmaxTask)
        assert task.kind == "softmax"
        assert task.trigger_slot == 5
        assert task.input_slot == 5
        assert task.output_slot == 6

    def test_round_trip_mat_mul_task(self) -> None:
        program = RuntimeProgram(
            version=1,
            mesh_config=MeshProgramConfig(width=2, height=1),
            pe_programs=[
                PEProgram(
                    coord=(0, 0),
                    tasks=[
                        MatMulTask(
                            trigger_slot=0,
                            matrix_slot=1,
                            vector_slot=0,
                            rows=3,
                            cols=2,
                            transpose=False,
                            output_slot=3,
                            routes=[
                                BroadcastRouteTask(dest=(1, 0), payload_slot=0),
                            ],
                        ),
                    ],
                    initial_sram={},
                ),
            ],
            input_slots=[],
        )
        restored = deserialize(serialize(program))

        task = restored.pe_programs[0].tasks[0]
        assert isinstance(task, MatMulTask)
        assert task.kind == "mat_mul"
        assert task.matrix_slot == 1
        assert task.vector_slot == 0
        assert task.rows == 3
        assert task.cols == 2
        assert task.transpose is False
        assert task.output_slot == 3
        assert len(task.routes) == 1
        assert task.routes[0].dest == (1, 0)


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
