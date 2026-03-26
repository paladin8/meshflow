"""Transformer block model helper — constructs GraphIR for a single-head transformer."""

import numpy as np

from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType


def transformer_block(seq_len: int, d_model: int, d_ff: int, eps: float = 1e-6) -> GraphIR:
    """Construct a single-head transformer block GraphIR.

    Block structure:
        Input → FORWARD → RMSNorm1 → Q/K/V proj → attention → out_proj → Add1
                  │                                                         ↑
                  └──────────────── (skip) ─────────────────────────────────┘
                Add1 → RMSNorm2 → FFN1 → ReLU → FFN2 → Add2
                  │                                       ↑
                  └──────────── (skip) ───────────────────┘
                Add2 → COLLECT
    """
    nodes = [
        Node(id="input", op=OpType.FORWARD),
        Node(id="rn1", op=OpType.RMSNORM, attrs={"eps": eps, "feature_count": d_model}),
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
        Node(id="add1", op=OpType.ADD),
        Node(id="rn2", op=OpType.RMSNORM, attrs={"eps": eps, "feature_count": d_model}),
        Node(
            id="ffn1",
            op=OpType.LINEAR,
            attrs={"in_features": d_model, "out_features": d_ff},
        ),
        Node(id="relu1", op=OpType.RELU),
        Node(
            id="ffn2",
            op=OpType.LINEAR,
            attrs={"in_features": d_ff, "out_features": d_model},
        ),
        Node(id="add2", op=OpType.ADD),
        Node(id="output", op=OpType.COLLECT),
    ]

    edges = [
        # Input → RMSNorm1 (main path) + Add1 (skip connection)
        Edge(src_node="input", src_slot=0, dst_node="rn1", dst_slot=0),
        Edge(src_node="input", src_slot=0, dst_node="add1", dst_slot=1),
        # RMSNorm1 → Q/K/V projections
        Edge(src_node="rn1", src_slot=0, dst_node="q_proj", dst_slot=0),
        Edge(src_node="rn1", src_slot=0, dst_node="k_proj", dst_slot=0),
        Edge(src_node="rn1", src_slot=0, dst_node="v_proj", dst_slot=0),
        # Q/K/V → attention chain
        Edge(src_node="q_proj", src_slot=0, dst_node="qkt", dst_slot=0),
        Edge(src_node="k_proj", src_slot=0, dst_node="qkt", dst_slot=1),
        Edge(src_node="v_proj", src_slot=0, dst_node="av", dst_slot=2),
        # Attention chain internal
        Edge(src_node="qkt", src_slot=0, dst_node="sm", dst_slot=0),
        Edge(src_node="sm", src_slot=0, dst_node="av", dst_slot=0),
        # AV → output projection
        Edge(src_node="av", src_slot=0, dst_node="out_proj", dst_slot=0),
        # Output projection → Add1 (main path, slot 0)
        Edge(src_node="out_proj", src_slot=0, dst_node="add1", dst_slot=0),
        # Add1 → RMSNorm2 (main path) + Add2 (skip connection)
        Edge(src_node="add1", src_slot=0, dst_node="rn2", dst_slot=0),
        Edge(src_node="add1", src_slot=0, dst_node="add2", dst_slot=1),
        # RMSNorm2 → FFN
        Edge(src_node="rn2", src_slot=0, dst_node="ffn1", dst_slot=0),
        # FFN chain
        Edge(src_node="ffn1", src_slot=0, dst_node="relu1", dst_slot=0),
        Edge(src_node="relu1", src_slot=0, dst_node="ffn2", dst_slot=0),
        # FFN2 → Add2 (main path, slot 0)
        Edge(src_node="ffn2", src_slot=0, dst_node="add2", dst_slot=0),
        # Add2 → output
        Edge(src_node="add2", src_slot=0, dst_node="output", dst_slot=0),
    ]

    return GraphIR(nodes=nodes, edges=edges)


def transformer_weights(
    d_model: int,
    d_ff: int,
    seed: int = 0,
) -> dict[str, dict[str, np.ndarray]]:
    """Generate random weights for a transformer block.

    Returns weights keyed by node ID, matching the graph from transformer_block().
    Uses Xavier-like initialization scaled for small test dimensions.
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(d_model)

    def linear_weights(in_f: int, out_f: int) -> dict[str, np.ndarray]:
        return {
            "weight": (rng.standard_normal((out_f, in_f)) * scale).astype(np.float32),
            "bias": np.zeros(out_f, dtype=np.float32),
        }

    return {
        "rn1": {"gamma": np.ones(d_model, dtype=np.float32)},
        "q_proj": linear_weights(d_model, d_model),
        "k_proj": linear_weights(d_model, d_model),
        "v_proj": linear_weights(d_model, d_model),
        "out_proj": linear_weights(d_model, d_model),
        "rn2": {"gamma": np.ones(d_model, dtype=np.float32)},
        "ffn1": linear_weights(d_model, d_ff),
        "ffn2": linear_weights(d_ff, d_model),
    }
