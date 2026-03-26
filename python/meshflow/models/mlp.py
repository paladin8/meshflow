"""MLP model helper — constructs GraphIR for a multi-layer perceptron."""

import numpy as np

from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType


def mlp_block(layer_dims: list[int]) -> GraphIR:
    """Construct an MLP GraphIR from a list of layer dimensions.

    Example: mlp_block([4, 8, 4]) creates:
        FORWARD → Linear(4→8) → ReLU → Linear(8→4) → COLLECT

    General pattern for N layers:
        FORWARD → (Linear → ReLU) × (N-1) → Linear → COLLECT

    No ReLU after the final layer.

    Node IDs: "input", "linear0", "relu0", "linear1", ..., "output".
    Edges connect sequentially with src_slot=0, dst_slot=0.
    """
    if len(layer_dims) < 2:
        raise ValueError("layer_dims must have at least 2 entries (in and out dimensions)")

    nodes: list[Node] = [Node(id="input", op=OpType.FORWARD)]

    num_layers = len(layer_dims) - 1
    for i in range(num_layers):
        nodes.append(
            Node(
                id=f"linear{i}",
                op=OpType.LINEAR,
                attrs={"in_features": layer_dims[i], "out_features": layer_dims[i + 1]},
            )
        )
        if i < num_layers - 1:
            nodes.append(Node(id=f"relu{i}", op=OpType.RELU))

    nodes.append(Node(id="output", op=OpType.COLLECT))

    # Build edges sequentially
    edges: list[Edge] = []
    prev_id = "input"
    for i in range(num_layers):
        linear_id = f"linear{i}"
        edges.append(Edge(src_node=prev_id, src_slot=0, dst_node=linear_id, dst_slot=0))
        prev_id = linear_id
        if i < num_layers - 1:
            relu_id = f"relu{i}"
            edges.append(Edge(src_node=prev_id, src_slot=0, dst_node=relu_id, dst_slot=0))
            prev_id = relu_id

    edges.append(Edge(src_node=prev_id, src_slot=0, dst_node="output", dst_slot=0))

    return GraphIR(nodes=nodes, edges=edges)


def mlp_weights(layer_dims: list[int], seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
    """Generate random weights for an MLP block.

    Returns weights keyed by node ID, matching the graph from mlp_block().
    Uses Xavier-like initialization (scale = 1/sqrt(in_features)).
    Zero biases.
    """
    if len(layer_dims) < 2:
        raise ValueError("layer_dims must have at least 2 entries (in and out dimensions)")

    rng = np.random.default_rng(seed)
    num_layers = len(layer_dims) - 1

    weights: dict[str, dict[str, np.ndarray]] = {}
    for i in range(num_layers):
        in_f = layer_dims[i]
        out_f = layer_dims[i + 1]
        scale = 1.0 / np.sqrt(in_f)
        weights[f"linear{i}"] = {
            "weight": (rng.standard_normal((out_f, in_f)) * scale).astype(np.float32),
            "bias": np.zeros(out_f, dtype=np.float32),
        }

    return weights
