"""Graph IR — pure topology, no placement information."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any


class OpType(Enum):
    FORWARD = "forward"
    COLLECT = "collect"
    LINEAR = "linear"
    RELU = "relu"

    @property
    def is_activation(self) -> bool:
        """Whether this op type is an activation function (fusable onto collect PEs)."""
        return self in _ACTIVATION_OPS


_ACTIVATION_OPS = frozenset({OpType.RELU})


@dataclass
class Node:
    id: str
    op: OpType
    attrs: dict[str, Any] | None = None


@dataclass
class Edge:
    src_node: str
    src_slot: int
    dst_node: str
    dst_slot: int


@dataclass
class GraphIR:
    nodes: list[Node]
    edges: list[Edge]

    def validate(self) -> None:
        """Validate graph structure. Raises ValueError on invalid graphs."""
        self._check_duplicate_node_ids()
        self._check_edge_references()
        self._check_acyclic()
        self._check_linear_attrs()
        self._check_activation_connectivity()

    def _check_linear_attrs(self) -> None:
        for node in self.nodes:
            if node.op == OpType.LINEAR:
                if node.attrs is None:
                    raise ValueError(
                        f"LINEAR node {node.id!r} requires attrs with in_features and out_features"
                    )
                for key in ("in_features", "out_features"):
                    if key not in node.attrs:
                        raise ValueError(f"LINEAR node {node.id!r} missing required attr: {key!r}")
                    if not isinstance(node.attrs[key], int) or node.attrs[key] <= 0:
                        raise ValueError(
                            f"LINEAR node {node.id!r} attr {key!r} must be a positive integer"
                        )

    def input_node_ids(self) -> list[str]:
        """Return IDs of nodes with no incoming edges (input entry points)."""
        nodes_with_incoming: set[str] = set()
        for edge in self.edges:
            nodes_with_incoming.add(edge.dst_node)
        return [n.id for n in self.nodes if n.id not in nodes_with_incoming]

    def topological_order(self) -> list[str]:
        """Return node IDs in topological order. Raises ValueError if cyclic."""
        node_ids = {n.id for n in self.nodes}
        adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
        in_degree: dict[str, int] = {nid: 0 for nid in node_ids}

        for edge in self.edges:
            adj[edge.src_node].append(edge.dst_node)
            in_degree[edge.dst_node] += 1

        # Process roots in node-list order for deterministic output
        queue = deque(n.id for n in self.nodes if in_degree[n.id] == 0)

        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(node_ids):
            raise ValueError("graph contains a cycle")

        return order

    def _check_duplicate_node_ids(self) -> None:
        seen: set[str] = set()
        for node in self.nodes:
            if node.id in seen:
                raise ValueError(f"duplicate node id: {node.id!r}")
            seen.add(node.id)

    def _check_edge_references(self) -> None:
        node_ids = {n.id for n in self.nodes}
        for edge in self.edges:
            if edge.src_node not in node_ids:
                raise ValueError(f"edge references unknown source node: {edge.src_node!r}")
            if edge.dst_node not in node_ids:
                raise ValueError(f"edge references unknown destination node: {edge.dst_node!r}")

    def _check_activation_connectivity(self) -> None:
        node_map = {n.id: n for n in self.nodes}
        for node in self.nodes:
            if not node.op.is_activation:
                continue
            kind = node.op.value.upper()
            # Must have exactly one incoming edge
            incoming = [e for e in self.edges if e.dst_node == node.id]
            if len(incoming) != 1:
                raise ValueError(
                    f"{kind} node {node.id!r} must have exactly one incoming edge, "
                    f"got {len(incoming)}"
                )
            # Incoming must be from a LINEAR node
            src = node_map[incoming[0].src_node]
            if src.op != OpType.LINEAR:
                raise ValueError(
                    f"{kind} node {node.id!r} must follow a LINEAR node, "
                    f"but predecessor {src.id!r} is {src.op.value}"
                )
            # Must have zero or one outgoing edge
            outgoing = [e for e in self.edges if e.src_node == node.id]
            if len(outgoing) > 1:
                raise ValueError(
                    f"{kind} node {node.id!r} must have at most one outgoing edge, "
                    f"got {len(outgoing)}"
                )

    def _check_acyclic(self) -> None:
        self.topological_order()  # raises ValueError if cyclic
