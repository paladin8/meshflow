"""End-to-end tests for multi-position (num_positions) support."""

import numpy as np
import torch

from meshflow._mesh_runtime import run_program
from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.artifact import serialize
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.models.reference import reference_linear, reference_rmsnorm


class TestBatchedLinear:
    """LINEAR with multi-position input through compile → run → validate."""

    def test_identity_two_positions(self) -> None:
        """Identity weights, 2 positions: output should match input."""
        in_f, out_f = 4, 4
        W = np.eye(in_f, dtype=np.float32)
        b = np.zeros(out_f, dtype=np.float32)

        graph = GraphIR(
            nodes=[
                Node(
                    id="lin", op=OpType.LINEAR, attrs={"in_features": in_f, "out_features": out_f}
                ),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="lin", src_slot=0, dst_node="col", dst_slot=0)],
        )
        config = CompilerConfig(mesh_height=3)  # 2 tiles + 1 collect
        program = compile(graph, config, weights={"lin": {"weight": W, "bias": b}})
        artifact_bytes = serialize(program)

        # 2 positions of 4 features each
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = run_program(artifact_bytes, inputs={"lin": x})

        # Find the collect PE output
        output = None
        for coord, data in result.outputs.items():
            output = data
        assert output is not None

        # Position-major output: [pos0_features, pos1_features]
        assert len(output) == 8
        assert all(abs(output[i] - x[i]) < 1e-5 for i in range(8))

    def test_matches_torch_two_positions(self) -> None:
        """Non-trivial weights, 2 positions, validate against torch."""
        torch.manual_seed(42)
        in_f, out_f = 4, 3

        W = torch.randn(out_f, in_f)
        b = torch.randn(out_f)
        x = torch.randn(2, in_f)  # 2 positions

        graph = GraphIR(
            nodes=[
                Node(
                    id="lin", op=OpType.LINEAR, attrs={"in_features": in_f, "out_features": out_f}
                ),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="lin", src_slot=0, dst_node="col", dst_slot=0)],
        )
        config = CompilerConfig(mesh_height=3)
        program = compile(graph, config, weights={"lin": {"weight": W.numpy(), "bias": b.numpy()}})
        artifact_bytes = serialize(program)

        result = run_program(artifact_bytes, inputs={"lin": x.flatten().tolist()})

        output = None
        for data in result.outputs.values():
            output = data
        assert output is not None

        # Per-position reference
        expected_0 = reference_linear(x[0], W, b)
        expected_1 = reference_linear(x[1], W, b)
        expected = torch.cat([expected_0, expected_1])

        actual = torch.tensor(output)
        assert torch.allclose(actual, expected, atol=1e-5), f"got {actual} vs expected {expected}"


class TestMultiPositionRmsNorm:
    """RMSNorm with multi-position input through compile → run → validate."""

    def test_single_position(self) -> None:
        """Single position RMSNorm validates against torch reference."""
        torch.manual_seed(42)
        d_model = 4
        eps = 1e-6

        gamma = torch.ones(d_model)
        x = torch.tensor([3.0, 4.0, 0.0, 0.0])

        graph = GraphIR(
            nodes=[
                Node(id="fwd", op=OpType.FORWARD),
                Node(
                    id="rn",
                    op=OpType.RMSNORM,
                    attrs={"eps": eps, "feature_count": d_model},
                ),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="fwd", src_slot=0, dst_node="rn", dst_slot=0),
                Edge(src_node="rn", src_slot=0, dst_node="col", dst_slot=0),
            ],
        )
        config = CompilerConfig(mesh_height=4)  # 2 tiles + reduce + collect
        weights = {"rn": {"gamma": gamma.numpy()}}
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        result = run_program(artifact_bytes, inputs={"fwd": x.tolist()})

        output = None
        for data in result.outputs.values():
            output = data
        assert output is not None

        expected = reference_rmsnorm(x, gamma, eps)
        actual = torch.tensor(output)
        assert torch.allclose(actual, expected, atol=1e-4), f"got {actual} vs expected {expected}"

    def test_two_positions(self) -> None:
        """Two-position RMSNorm validates per-position against torch reference."""
        torch.manual_seed(42)
        d_model = 4
        eps = 1e-6

        gamma = torch.ones(d_model)
        x = torch.tensor([[3.0, 4.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        graph = GraphIR(
            nodes=[
                Node(id="fwd", op=OpType.FORWARD),
                Node(
                    id="rn",
                    op=OpType.RMSNORM,
                    attrs={"eps": eps, "feature_count": d_model},
                ),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="fwd", src_slot=0, dst_node="rn", dst_slot=0),
                Edge(src_node="rn", src_slot=0, dst_node="col", dst_slot=0),
            ],
        )
        # 2 tiles + 1 reduce + 1 collect = 4 rows; mesh_height=4 → min(4-2,4) = 2 tiles
        config = CompilerConfig(mesh_height=4)
        weights = {"rn": {"gamma": gamma.numpy()}}
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        result = run_program(artifact_bytes, inputs={"fwd": x.flatten().tolist()})

        output = None
        for data in result.outputs.values():
            output = data
        assert output is not None

        expected = reference_rmsnorm(x, gamma, eps)
        actual = torch.tensor(output).reshape(2, d_model)
        assert torch.allclose(actual, expected, atol=1e-4), f"got {actual} vs expected {expected}"


class TestAttentionMatMulChain:
    """Q/K/V projections → MATMUL(QK^T) → Softmax → MATMUL(AV) through compile → run → validate."""

    def test_seq_len_1_identity_projections(self) -> None:
        """seq_len=1, identity Q/K/V/out projections.

        With identity projections: Q=K=V=x, QK^T=dot(x,x)=scalar,
        softmax([scalar])=[1.0], AV=1.0*V=V, out_proj(V)=V=x.
        So attention output should equal input.
        """
        d_model = 4
        seq_len = 1

        # Build graph: input → Q/K/V proj → attention → out_proj → collect
        graph = GraphIR(
            nodes=[
                Node(id="input", op=OpType.FORWARD),
                Node(
                    id="q_proj",
                    op=OpType.LINEAR,
                    attrs={"in_features": d_model, "out_features": d_model},
                ),
                Node(
                    id="k_proj",
                    op=OpType.LINEAR,
                    attrs={"in_features": d_model, "out_features": d_model},
                ),
                Node(
                    id="v_proj",
                    op=OpType.LINEAR,
                    attrs={"in_features": d_model, "out_features": d_model},
                ),
                Node(id="qkt", op=OpType.MATMUL, attrs={"seq_len": seq_len, "d_model": d_model}),
                Node(id="sm", op=OpType.SOFTMAX),
                Node(id="av", op=OpType.MATMUL, attrs={"seq_len": seq_len, "d_model": d_model}),
                Node(
                    id="out_proj",
                    op=OpType.LINEAR,
                    attrs={"in_features": d_model, "out_features": d_model},
                ),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                # Input broadcasts to Q/K/V projections
                Edge(src_node="input", src_slot=0, dst_node="q_proj", dst_slot=0),
                Edge(src_node="input", src_slot=1, dst_node="k_proj", dst_slot=0),
                Edge(src_node="input", src_slot=2, dst_node="v_proj", dst_slot=0),
                # Q collect → attention PE slot 0 (Q vector)
                Edge(src_node="q_proj", src_slot=0, dst_node="qkt", dst_slot=0),
                # K collect → attention PE slot 1 (K matrix)
                Edge(src_node="k_proj", src_slot=0, dst_node="qkt", dst_slot=1),
                # V collect → attention PE slot 2 (V matrix)
                Edge(src_node="v_proj", src_slot=0, dst_node="av", dst_slot=2),
                # Attention chain
                Edge(src_node="qkt", src_slot=0, dst_node="sm", dst_slot=0),
                Edge(src_node="sm", src_slot=0, dst_node="av", dst_slot=0),
                # AV → out proj
                Edge(src_node="av", src_slot=0, dst_node="out_proj", dst_slot=0),
                # Out proj → collect
                Edge(src_node="out_proj", src_slot=0, dst_node="col", dst_slot=0),
            ],
        )

        # Identity weights for all projections
        W_id = np.eye(d_model, dtype=np.float32)
        b_zero = np.zeros(d_model, dtype=np.float32)
        weights = {
            "q_proj": {"weight": W_id, "bias": b_zero},
            "k_proj": {"weight": W_id, "bias": b_zero},
            "v_proj": {"weight": W_id, "bias": b_zero},
            "out_proj": {"weight": W_id, "bias": b_zero},
        }

        config = CompilerConfig(mesh_height=3)

        program = compile(graph, config, weights=weights)

        artifact_bytes = serialize(program)

        x = [1.0, 2.0, 3.0, 4.0]
        result = run_program(artifact_bytes, inputs={"input": x})

        # Find output (at the collect PE)
        output = None
        for data in result.outputs.values():
            output = data
        assert output is not None, f"no output found, outputs={dict(result.outputs)}"

        # With identity projections and seq_len=1: output should equal input
        actual = torch.tensor(output)
        expected = torch.tensor(x)
        assert torch.allclose(actual, expected, atol=1e-4), f"got {actual} vs expected {expected}"

    def test_seq_len_4_with_torch_validation(self) -> None:
        """seq_len=4, d_model=4, random projections, validate against torch."""
        torch.manual_seed(42)
        d_model = 4
        seq_len = 4

        W_q = torch.randn(d_model, d_model)
        b_q = torch.randn(d_model)
        W_k = torch.randn(d_model, d_model)
        b_k = torch.randn(d_model)
        W_v = torch.randn(d_model, d_model)
        b_v = torch.randn(d_model)
        W_o = torch.randn(d_model, d_model)
        b_o = torch.randn(d_model)

        # Input: seq_len positions of d_model features
        x = torch.randn(seq_len, d_model)

        # Torch reference: per-position projections → attention → output projection
        Q = torch.nn.functional.linear(x, W_q, b_q)  # (seq_len, d_model)
        K = torch.nn.functional.linear(x, W_k, b_k)
        V = torch.nn.functional.linear(x, W_v, b_v)
        scores = Q @ K.T  # (seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = attn_weights @ V  # (seq_len, d_model)
        expected = torch.nn.functional.linear(attn_out, W_o, b_o)  # (seq_len, d_model)

        # Build graph
        graph = GraphIR(
            nodes=[
                Node(id="input", op=OpType.FORWARD),
                Node(
                    id="q_proj",
                    op=OpType.LINEAR,
                    attrs={"in_features": d_model, "out_features": d_model},
                ),
                Node(
                    id="k_proj",
                    op=OpType.LINEAR,
                    attrs={"in_features": d_model, "out_features": d_model},
                ),
                Node(
                    id="v_proj",
                    op=OpType.LINEAR,
                    attrs={"in_features": d_model, "out_features": d_model},
                ),
                Node(id="qkt", op=OpType.MATMUL, attrs={"seq_len": seq_len, "d_model": d_model}),
                Node(id="sm", op=OpType.SOFTMAX),
                Node(id="av", op=OpType.MATMUL, attrs={"seq_len": seq_len, "d_model": d_model}),
                Node(
                    id="out_proj",
                    op=OpType.LINEAR,
                    attrs={"in_features": d_model, "out_features": d_model},
                ),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="input", src_slot=0, dst_node="q_proj", dst_slot=0),
                Edge(src_node="input", src_slot=1, dst_node="k_proj", dst_slot=0),
                Edge(src_node="input", src_slot=2, dst_node="v_proj", dst_slot=0),
                Edge(src_node="q_proj", src_slot=0, dst_node="qkt", dst_slot=0),
                Edge(src_node="k_proj", src_slot=0, dst_node="qkt", dst_slot=1),
                Edge(src_node="v_proj", src_slot=0, dst_node="av", dst_slot=2),
                Edge(src_node="qkt", src_slot=0, dst_node="sm", dst_slot=0),
                Edge(src_node="sm", src_slot=0, dst_node="av", dst_slot=0),
                Edge(src_node="av", src_slot=0, dst_node="out_proj", dst_slot=0),
                Edge(src_node="out_proj", src_slot=0, dst_node="col", dst_slot=0),
            ],
        )

        weights = {
            "q_proj": {"weight": W_q.numpy(), "bias": b_q.numpy()},
            "k_proj": {"weight": W_k.numpy(), "bias": b_k.numpy()},
            "v_proj": {"weight": W_v.numpy(), "bias": b_v.numpy()},
            "out_proj": {"weight": W_o.numpy(), "bias": b_o.numpy()},
        }

        # Need enough mesh height for: LINEAR tiles + collect + attention PEs + collect
        config = CompilerConfig(mesh_height=6)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        output = None
        for data in result.outputs.values():
            output = data
        assert output is not None, f"no output found, outputs={dict(result.outputs)}"

        actual = torch.tensor(output).reshape(seq_len, d_model)
        assert torch.allclose(actual, expected, atol=1e-3), (
            f"got {actual}\nvs expected {expected}\ndiff={actual - expected}"
        )
