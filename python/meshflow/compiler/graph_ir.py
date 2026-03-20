"""Graph IR — pure topology, no placement information."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any


class OpType(Enum):
    FORWARD = "forward"
    COLLECT = "collect"
    LINEAR = "linear"


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

    def _check_acyclic(self) -> None:
        self.topological_order()  # raises ValueError if cyclic
