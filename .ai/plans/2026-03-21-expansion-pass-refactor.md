# Expansion Pass Refactor — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate operator expansion from coordinate assignment by introducing an ExpandedIR and expansion pass between GraphIR and SpatialIR, replacing untyped `attrs` dicts with typed data on placed nodes and making inter-layer edges explicit.

**Architecture:** A new `expand` pass transforms GraphIR into ExpandedIR (operator groups with typed tile/collect specs and explicit inter-group links). Placement consumes ExpandedIR to assign coordinates, producing SpatialIR with typed `PlacedNodeData` and explicit inter-group edges. Routing reads typed data and edges uniformly — no more `tiles_by_origin` index or `route_to` attr side-channel.

**Tech Stack:** Python 3.12, dataclasses, existing meshflow compiler infrastructure.

---

## File Structure

**New files:**
- `python/meshflow/compiler/expanded_ir.py` — ExpandedIR data types (TileSpec, CollectSpec, TiledComputeGroup, ExpandedIR)
- `python/meshflow/compiler/passes/expand.py` — Expansion pass (GraphIR + config → ExpandedIR)
- `tests/python/compiler/test_expanded_ir.py` — ExpandedIR type construction tests
- `tests/python/compiler/passes/test_expand.py` — Expansion pass tests

**Modified files:**
- `python/meshflow/compiler/spatial_ir.py` — Add PlacedTileData, PlacedCollectData, PlacedNodeData; add `data` field to PlacedNode (keep `attrs` temporarily for routing backward compat, removed in Task 5)
- `python/meshflow/compiler/passes/place.py` — Consume ExpandedIR, produce typed data + explicit inter-group edges
- `python/meshflow/compiler/passes/route.py` — Read typed `node.data`, use explicit edges for inter-group routing
- `python/meshflow/compiler/passes/__init__.py` — Export `expand`
- `python/meshflow/compiler/__init__.py` — Wire `expand` into pipeline before `place`
- `tests/python/compiler/passes/test_place.py` — Call `expand()` before `place()`, assert typed data
- `tests/python/compiler/passes/test_route.py` — Construct SpatialIR with typed data, update assertions
- `tests/python/compiler/test_compile.py` — Minor updates if needed

**Untouched files:**
- `graph_ir.py`, `schedule_ir.py`, `artifact.py`, `config.py`, `passes/lower.py` — no changes needed
- `crates/mesh_runtime/` — no Rust changes (artifact format unchanged)
- `tests/python/runtime/test_end_to_end.py` — should pass unchanged (tests go through `compile()`)

---

## Chunk 1: ExpandedIR + Expand Pass + Typed SpatialIR + Placement Refactor

### Task 1: ExpandedIR Data Types

**Files:**
- Create: `python/meshflow/compiler/expanded_ir.py`
- Test: `tests/python/compiler/test_expanded_ir.py`

- [ ] **Step 1: Write tests for ExpandedIR data types**

```python
# tests/python/compiler/test_expanded_ir.py
"""Tests for ExpandedIR data types."""

from meshflow.compiler.expanded_ir import (
    CollectSpec,
    ExpandedIR,
    TiledComputeGroup,
    TileSpec,
)


class TestTileSpec:
    def test_construction(self):
        spec = TileSpec(tile_index=0, tile_rows=4, fragment_offset=0, in_features=8)
        assert spec.tile_index == 0
        assert spec.tile_rows == 4
        assert spec.fragment_offset == 0
        assert spec.in_features == 8

    def test_uneven_tile(self):
        spec = TileSpec(tile_index=2, tile_rows=3, fragment_offset=9, in_features=8)
        assert spec.tile_rows == 3
        assert spec.fragment_offset == 9


class TestCollectSpec:
    def test_construction(self):
        spec = CollectSpec(num_fragments=3, total_rows=10)
        assert spec.num_fragments == 3
        assert spec.total_rows == 10
        assert spec.activation is None

    def test_with_activation(self):
        spec = CollectSpec(num_fragments=2, total_rows=8, activation="relu")
        assert spec.activation == "relu"


class TestTiledComputeGroup:
    def test_single_tile(self):
        group = TiledComputeGroup(
            origin_id="linear1",
            tiles=[TileSpec(tile_index=0, tile_rows=4, fragment_offset=0, in_features=8)],
            collect=CollectSpec(num_fragments=1, total_rows=4),
        )
        assert group.origin_id == "linear1"
        assert len(group.tiles) == 1
        assert group.next_group is None

    def test_with_next_group(self):
        group = TiledComputeGroup(
            origin_id="linear1",
            tiles=[TileSpec(tile_index=0, tile_rows=4, fragment_offset=0, in_features=8)],
            collect=CollectSpec(num_fragments=1, total_rows=4, activation="relu"),
            next_group="linear2",
        )
        assert group.next_group == "linear2"
        assert group.collect.activation == "relu"


class TestExpandedIR:
    def test_empty(self):
        ir = ExpandedIR(groups=[], passthrough_nodes=[], passthrough_edges=[])
        assert len(ir.groups) == 0

    def test_passthrough(self):
        from meshflow.compiler.graph_ir import Edge, Node, OpType

        nodes = [Node(id="fwd", op=OpType.FORWARD)]
        edges = [Edge(src_node="fwd", src_slot=0, dst_node="col", dst_slot=0)]
        ir = ExpandedIR(groups=[], passthrough_nodes=nodes, passthrough_edges=edges)
        assert len(ir.passthrough_nodes) == 1
        assert len(ir.passthrough_edges) == 1

    def test_with_groups(self):
        group = TiledComputeGroup(
            origin_id="linear1",
            tiles=[
                TileSpec(tile_index=0, tile_rows=2, fragment_offset=0, in_features=4),
                TileSpec(tile_index=1, tile_rows=2, fragment_offset=2, in_features=4),
            ],
            collect=CollectSpec(num_fragments=2, total_rows=4),
        )
        ir = ExpandedIR(groups=[group], passthrough_nodes=[], passthrough_edges=[])
        assert len(ir.groups) == 1
        assert ir.groups[0].collect.num_fragments == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/python/compiler/test_expanded_ir.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'meshflow.compiler.expanded_ir'`

- [ ] **Step 3: Implement ExpandedIR data types**

```python
# python/meshflow/compiler/expanded_ir.py
"""Expanded IR — operator nodes expanded into physical PE groups, before placement."""

from dataclasses import dataclass, field

from meshflow.compiler.graph_ir import Edge, Node


@dataclass
class TileSpec:
    """Specification for a single compute tile within a tiled operator group."""

    tile_index: int
    tile_rows: int
    fragment_offset: int
    in_features: int


@dataclass
class CollectSpec:
    """Specification for the collect PE that gathers tile fragments."""

    num_fragments: int
    total_rows: int
    activation: str | None = None


@dataclass
class TiledComputeGroup:
    """A LINEAR operator expanded into parallel compute tiles + a collect PE.

    Each tile computes a fragment of the output (y_i = W_i @ x + b_i).
    The collect PE gathers fragments into the full output vector.
    """

    origin_id: str
    tiles: list[TileSpec]
    collect: CollectSpec
    next_group: str | None = None


@dataclass
class ExpandedIR:
    """IR after operator expansion, before coordinate assignment.

    For LINEAR graphs: groups contain the tiled structure.
    For FORWARD/COLLECT graphs: passthrough_nodes/edges carry the original graph.
    """

    groups: list[TiledComputeGroup] = field(default_factory=list)
    passthrough_nodes: list[Node] = field(default_factory=list)
    passthrough_edges: list[Edge] = field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/python/compiler/test_expanded_ir.py -v`
Expected: PASS (all 8 tests)

---

### Task 2: Expand Pass

**Files:**
- Create: `python/meshflow/compiler/passes/expand.py`
- Modify: `python/meshflow/compiler/passes/__init__.py`
- Test: `tests/python/compiler/passes/test_expand.py`

- [ ] **Step 1: Write tests for the expand pass**

```python
# tests/python/compiler/passes/test_expand.py
"""Tests for the expansion pass."""

import pytest

from meshflow.compiler.config import CompilerConfig
from meshflow.compiler.expanded_ir import TiledComputeGroup
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.compiler.passes.expand import expand


class TestExpandPassthrough:
    """FORWARD/COLLECT graphs pass through unchanged."""

    def test_forward_collect_passthrough(self):
        graph = GraphIR(
            nodes=[
                Node(id="fwd", op=OpType.FORWARD),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="fwd", src_slot=0, dst_node="col", dst_slot=0)],
        )
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 0
        assert len(expanded.passthrough_nodes) == 2
        assert len(expanded.passthrough_edges) == 1

    def test_empty_graph(self):
        graph = GraphIR(nodes=[], edges=[])
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 0
        assert len(expanded.passthrough_nodes) == 0


class TestExpandSingleLinear:
    """Single LINEAR node expansion."""

    def test_single_linear_default_tiles(self):
        """Without mesh_height, each output row gets its own tile."""
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 3})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 1
        group = expanded.groups[0]
        assert group.origin_id == "lin"
        assert len(group.tiles) == 3
        assert group.collect.num_fragments == 3
        assert group.collect.total_rows == 3
        assert group.next_group is None

    def test_single_linear_with_mesh_height(self):
        """mesh_height limits tile count."""
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 8, "out_features": 6})],
            edges=[],
        )
        config = CompilerConfig(mesh_height=4)  # 3 tiles + 1 collect
        expanded = expand(graph, config)
        group = expanded.groups[0]
        assert len(group.tiles) == 3
        assert group.collect.num_fragments == 3
        assert group.collect.total_rows == 6

    def test_tile_rows_even_split(self):
        """Even split: 6 rows / 3 tiles = 2 rows each."""
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6})],
            edges=[],
        )
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        tiles = expanded.groups[0].tiles
        assert [t.tile_rows for t in tiles] == [2, 2, 2]
        assert [t.fragment_offset for t in tiles] == [0, 2, 4]

    def test_tile_rows_uneven_split(self):
        """Uneven split: 7 rows / 3 tiles = [3, 2, 2]."""
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 7})],
            edges=[],
        )
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        tiles = expanded.groups[0].tiles
        assert [t.tile_rows for t in tiles] == [3, 2, 2]
        assert [t.fragment_offset for t in tiles] == [0, 3, 5]

    def test_in_features_propagated(self):
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 16, "out_features": 4})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        for tile in expanded.groups[0].tiles:
            assert tile.in_features == 16

    def test_mesh_height_too_small(self):
        """mesh_height=1 leaves 0 tiles — should error."""
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4})],
            edges=[],
        )
        config = CompilerConfig(mesh_height=1)
        with pytest.raises(ValueError, match="need at least 1 tile"):
            expand(graph, config)


class TestExpandActivationFusion:
    """LINEAR→activation fusion."""

    def test_linear_relu_fused(self):
        graph = GraphIR(
            nodes=[
                Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4}),
                Node(id="relu", op=OpType.RELU),
            ],
            edges=[Edge(src_node="lin", src_slot=0, dst_node="relu", dst_slot=0)],
        )
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 1
        assert expanded.groups[0].collect.activation == "relu"

    def test_linear_no_activation(self):
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        assert expanded.groups[0].collect.activation is None


class TestExpandMultiLayer:
    """Multi-layer LINEAR chains."""

    def test_two_layer_chain(self):
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6}),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[Edge(src_node="l1", src_slot=0, dst_node="l2", dst_slot=0)],
        )
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 2
        assert expanded.groups[0].origin_id == "l1"
        assert expanded.groups[0].next_group == "l2"
        assert expanded.groups[1].origin_id == "l2"
        assert expanded.groups[1].next_group is None

    def test_three_layer_with_relu(self):
        """L1 → RELU → L2 → RELU → L3"""
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 6}),
                Node(id="r2", op=OpType.RELU),
                Node(id="l3", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
                Edge(src_node="l2", src_slot=0, dst_node="r2", dst_slot=0),
                Edge(src_node="r2", src_slot=0, dst_node="l3", dst_slot=0),
            ],
        )
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 3
        assert expanded.groups[0].collect.activation == "relu"
        assert expanded.groups[0].next_group == "l2"
        assert expanded.groups[1].collect.activation == "relu"
        assert expanded.groups[1].next_group == "l3"
        assert expanded.groups[2].collect.activation is None
        assert expanded.groups[2].next_group is None

    def test_groups_preserve_topological_order(self):
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        expanded = expand(graph, CompilerConfig())
        assert [g.origin_id for g in expanded.groups] == ["l1", "l2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/python/compiler/passes/test_expand.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'meshflow.compiler.passes.expand'`

- [ ] **Step 3: Implement the expand pass**

```python
# python/meshflow/compiler/passes/expand.py
"""Expansion pass — expands operator nodes into physical PE group structures."""

from meshflow.compiler.config import CompilerConfig
from meshflow.compiler.expanded_ir import (
    CollectSpec,
    ExpandedIR,
    TiledComputeGroup,
    TileSpec,
)
from meshflow.compiler.graph_ir import GraphIR, OpType


def expand(graph: GraphIR, config: CompilerConfig) -> ExpandedIR:
    """Expand graph operator nodes into physical PE groups.

    LINEAR nodes become TiledComputeGroups (tiles + collect).
    Activation nodes following LINEAR are fused onto the collect spec.
    FORWARD/COLLECT nodes pass through unchanged.
    """
    graph.validate()
    topo_order = graph.topological_order()
    node_map = {n.id: n for n in graph.nodes}

    has_linear = any(node_map[nid].op == OpType.LINEAR for nid in topo_order)

    if has_linear:
        return _expand_linear(graph, config, topo_order, node_map)
    else:
        # Preserve topological order for deterministic placement
        return ExpandedIR(
            passthrough_nodes=[node_map[nid] for nid in topo_order],
            passthrough_edges=list(graph.edges),
        )


def _expand_linear(
    graph: GraphIR,
    config: CompilerConfig,
    topo_order: list[str],
    node_map: dict,
) -> ExpandedIR:
    """Expand LINEAR graph into tiled compute groups."""
    # Detect LINEAR → activation fusion pairs
    fused_activations: set[str] = set()
    activation_for_linear: dict[str, str] = {}
    for nid in topo_order:
        node = node_map[nid]
        if node.op != OpType.LINEAR:
            continue
        outgoing = [e for e in graph.edges if e.src_node == nid]
        if len(outgoing) == 1:
            successor = node_map[outgoing[0].dst_node]
            if successor.op.is_activation:
                fused_activations.add(successor.id)
                activation_for_linear[nid] = successor.id

    groups: list[TiledComputeGroup] = []

    for nid in topo_order:
        node = node_map[nid]

        if nid in fused_activations:
            continue
        if node.op != OpType.LINEAR:
            continue

        if node.attrs is None:
            raise ValueError(f"LINEAR node {nid!r} requires attrs")
        out_f = node.attrs["out_features"]
        in_f = node.attrs["in_features"]

        # Determine tile count
        if config.mesh_height is not None:
            num_tiles = min(config.mesh_height - 1, out_f)
        else:
            num_tiles = out_f

        if num_tiles < 1:
            raise ValueError(
                f"LINEAR node {nid!r}: need at least 1 tile, "
                f"but mesh_height={config.mesh_height} is too small"
            )

        # Build tile specs
        tiles: list[TileSpec] = []
        base = out_f // num_tiles
        remainder = out_f % num_tiles
        for i in range(num_tiles):
            tile_rows = base + 1 if i < remainder else base
            frag_offset = i * base + min(i, remainder)
            tiles.append(
                TileSpec(
                    tile_index=i,
                    tile_rows=tile_rows,
                    fragment_offset=frag_offset,
                    in_features=in_f,
                )
            )

        # Activation from fused node
        activation = None
        if nid in activation_for_linear:
            act_node = node_map[activation_for_linear[nid]]
            activation = act_node.op.value

        collect = CollectSpec(
            num_fragments=num_tiles,
            total_rows=out_f,
            activation=activation,
        )

        # Determine next group: follow outgoing edges (possibly through fused activation)
        effective_src = activation_for_linear.get(nid, nid)
        next_outgoing = [e for e in graph.edges if e.src_node == effective_src]
        next_group = None
        if next_outgoing:
            next_node = node_map[next_outgoing[0].dst_node]
            if next_node.op == OpType.LINEAR:
                next_group = next_node.id

        groups.append(
            TiledComputeGroup(
                origin_id=nid,
                tiles=tiles,
                collect=collect,
                next_group=next_group,
            )
        )

    return ExpandedIR(groups=groups)
```

- [ ] **Step 4: Export expand from passes package**

Add to `python/meshflow/compiler/passes/__init__.py`:

```python
from meshflow.compiler.passes.expand import expand
```

(Alongside existing `place`, `route`, `lower` exports.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/python/compiler/passes/test_expand.py -v`
Expected: PASS (all 13 tests)

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `uv run pytest tests/python -v && cargo test -p mesh_runtime`
Expected: All existing tests pass (expand is a pure addition, nothing consumes it yet)

- [ ] **Step 7: Commit**

```bash
git add python/meshflow/compiler/expanded_ir.py python/meshflow/compiler/passes/expand.py \
    python/meshflow/compiler/passes/__init__.py \
    tests/python/compiler/test_expanded_ir.py tests/python/compiler/passes/test_expand.py
git commit -m "refactor: add ExpandedIR types and expansion pass

Introduces the expand pass between GraphIR and SpatialIR.
LINEAR nodes are expanded into TiledComputeGroup structures with
typed TileSpec and CollectSpec. Activation fusion and inter-group
chaining are handled here. FORWARD/COLLECT graphs pass through.

Pure addition — not yet wired into the compiler pipeline."
```

---

### Task 3: Typed SpatialIR + Placement Refactor

**Files:**
- Modify: `python/meshflow/compiler/spatial_ir.py`
- Modify: `python/meshflow/compiler/passes/place.py`
- Modify: `python/meshflow/compiler/__init__.py`
- Modify: `tests/python/compiler/passes/test_place.py`
- Modify: `tests/python/compiler/passes/test_route.py` (call expand() before place())
- Modify: `tests/python/compiler/test_compile.py`

This task changes `place()` to consume ExpandedIR instead of GraphIR, adds typed `PlacedNodeData` alongside the existing `attrs` field (for routing backward compat), and emits explicit inter-group edges. The `attrs` field is kept temporarily so the routing pass continues to work unchanged — it will be removed in Task 5 after routing is updated.

- [ ] **Step 1: Update placement tests (TDD — tests first)**

Update `tests/python/compiler/passes/test_place.py`. Every test needs two changes:

1. Add imports at top of file:
   ```python
   from meshflow.compiler.passes.expand import expand
   from meshflow.compiler.spatial_ir import PlacedCollectData, PlacedTileData
   ```

2. Every call to `place(graph, config)` becomes:
   ```python
   expanded = expand(graph, config)
   spatial = place(expanded, config)
   ```

3. Add typed data assertions alongside existing attrs assertions. For LINEAR tile nodes:
   ```python
   assert isinstance(tile.data, PlacedTileData)
   assert tile.data.tile_rows == 2
   assert tile.data.origin_id == "lin"
   ```

4. For COLLECT nodes from LINEAR expansion:
   ```python
   assert isinstance(collect.data, PlacedCollectData)
   assert collect.data.num_fragments == 2
   assert collect.data.total_rows == 4
   ```

5. For FORWARD/COLLECT passthrough nodes:
   ```python
   assert node.data is None
   ```

6. For multi-layer tests: add assertions that explicit collect→tile inter-group edges exist. Invert `test_no_inter_layer_placed_edges` — it should now assert these edges ARE present:
   ```python
   collect_to_tile_edges = [e for e in spatial.edges if e.src_node == "l1_collect"]
   assert len(collect_to_tile_edges) == expected_num_next_tiles
   ```

- [ ] **Step 2: Update routing tests to call expand() before place()**

In `tests/python/compiler/passes/test_route.py`, every test that calls `place(graph, config)` needs the same signature change. The routing tests don't need typed data assertions yet (that's Task 4) — they just need to construct the SpatialIR correctly.

1. Add import at top:
   ```python
   from meshflow.compiler.passes.expand import expand
   ```

2. Every `place(graph, config)` becomes:
   ```python
   expanded = expand(graph, config)
   spatial = place(expanded, config)
   ```

No other changes to routing tests — they still assert on routing output (ScheduleIR), not on SpatialIR node data.

- [ ] **Step 3: Run updated tests to verify they fail**

Run: `uv run pytest tests/python/compiler/passes/test_place.py tests/python/compiler/passes/test_route.py -v`
Expected: FAIL — `expand` doesn't exist yet in passes, `place()` still takes GraphIR, PlacedNode has no `data` field.

- [ ] **Step 4: Add typed data classes to SpatialIR**

Keep `attrs` temporarily for routing backward compat. Add `data` alongside it:

```python
# python/meshflow/compiler/spatial_ir.py
"""Spatial IR — graph with placement (each node assigned a PE coordinate)."""

from dataclasses import dataclass
from typing import Any

from meshflow.compiler.graph_ir import OpType


@dataclass
class PlacedTileData:
    """Typed data for a LINEAR tile PE."""

    tile_index: int
    tile_rows: int
    fragment_offset: int
    in_features: int
    origin_id: str


@dataclass
class PlacedCollectData:
    """Typed data for a COLLECT PE from a tiled operator group."""

    num_fragments: int
    total_rows: int
    origin_id: str
    activation: str | None = None


PlacedNodeData = PlacedTileData | PlacedCollectData | None


@dataclass
class PlacedNode:
    id: str
    op: OpType
    coord: tuple[int, int]
    data: PlacedNodeData = None
    # DEPRECATED: attrs is kept temporarily for routing backward compat.
    # Will be removed in Task 5 after routing is updated to use typed data.
    attrs: dict[str, Any] | None = None


@dataclass
class PlacedEdge:
    src_node: str
    src_slot: int
    dst_node: str
    dst_slot: int


@dataclass
class SpatialIR:
    width: int
    height: int
    nodes: list[PlacedNode]
    edges: list[PlacedEdge]
```

- [ ] **Step 5: Rewrite placement pass to consume ExpandedIR**

Placement produces both `data` (typed) and `attrs` (dict, for routing backward compat).

```python
# python/meshflow/compiler/passes/place.py
"""Placement pass — assigns each expanded node/group to a PE coordinate."""

from meshflow.compiler.config import CompilerConfig, PlacementStrategy
from meshflow.compiler.expanded_ir import ExpandedIR
from meshflow.compiler.graph_ir import OpType
from meshflow.compiler.spatial_ir import (
    PlacedCollectData,
    PlacedEdge,
    PlacedNode,
    PlacedTileData,
    SpatialIR,
)


def place(expanded: ExpandedIR, config: CompilerConfig) -> SpatialIR:
    """Place expanded nodes/groups onto a 2D mesh grid.

    Dispatches on config.placement strategy.
    """
    if config.placement == PlacementStrategy.SEQUENTIAL:
        return _place_sequential(expanded, config)

    raise ValueError(f"unknown placement strategy: {config.placement!r}")


def _place_sequential(expanded: ExpandedIR, config: CompilerConfig) -> SpatialIR:
    """Sequential placement.

    Tiled compute groups use vertical column layout (one column per group).
    Passthrough nodes use horizontal row-major layout (M2 compat).
    """
    if expanded.groups:
        return _place_linear_columns(expanded, config)
    else:
        return _place_row_major(expanded, config)


def _place_row_major(expanded: ExpandedIR, config: CompilerConfig) -> SpatialIR:
    """Row-major placement for passthrough (FORWARD/COLLECT) graphs."""
    total = len(expanded.passthrough_nodes)
    if total == 0:
        width = config.mesh_width or 1
        height = config.mesh_height or 1
    else:
        width = config.mesh_width or total
        height = config.mesh_height or 1

    placed_nodes: list[PlacedNode] = []
    placed_edges: list[PlacedEdge] = []

    for idx, node in enumerate(expanded.passthrough_nodes):
        x = idx % width
        y = idx // width
        if x >= width or y >= height:
            raise ValueError(
                f"node {node.id!r} at index {idx} does not fit in {width}x{height} mesh"
            )
        placed_nodes.append(PlacedNode(id=node.id, op=node.op, coord=(x, y)))

    for e in expanded.passthrough_edges:
        placed_edges.append(
            PlacedEdge(
                src_node=e.src_node,
                src_slot=e.src_slot,
                dst_node=e.dst_node,
                dst_slot=e.dst_slot,
            )
        )

    return SpatialIR(width=width, height=height, nodes=placed_nodes, edges=placed_edges)


def _place_linear_columns(expanded: ExpandedIR, config: CompilerConfig) -> SpatialIR:
    """Column-per-group placement for tiled compute groups.

    Each group gets one column. Tiles stack vertically (y-axis).
    Collect PE sits at the top of each column.
    Explicit edges connect collect → next group's tiles for inter-layer routing.
    """
    placed_nodes: list[PlacedNode] = []
    placed_edges: list[PlacedEdge] = []

    # Track placed nodes per group for inter-group edge generation
    group_tile_nodes: dict[str, list[PlacedNode]] = {}
    group_collect_node: dict[str, PlacedNode] = {}

    col = 0
    max_column_height = 0

    for group in expanded.groups:
        tile_nodes: list[PlacedNode] = []

        for spec in group.tiles:
            tile_id = f"{group.origin_id}_tile_{spec.tile_index}"
            # Build attrs dict for routing backward compat (removed in Task 5)
            tile_attrs = {
                "tile_index": spec.tile_index,
                "tile_rows": spec.tile_rows,
                "fragment_offset": spec.fragment_offset,
                "in_features": spec.in_features,
                "origin_id": group.origin_id,
            }
            node = PlacedNode(
                id=tile_id,
                op=OpType.LINEAR,
                coord=(col, spec.tile_index),
                data=PlacedTileData(
                    tile_index=spec.tile_index,
                    tile_rows=spec.tile_rows,
                    fragment_offset=spec.fragment_offset,
                    in_features=spec.in_features,
                    origin_id=group.origin_id,
                ),
                attrs=tile_attrs,
            )
            placed_nodes.append(node)
            tile_nodes.append(node)

        # Collect PE at top of column
        collect_row = len(group.tiles)
        collect_id = f"{group.origin_id}_collect"

        # Build collect attrs for routing backward compat (removed in Task 5)
        collect_attrs: dict = {
            "num_fragments": group.collect.num_fragments,
            "total_rows": group.collect.total_rows,
            "origin_id": group.origin_id,
        }
        if group.collect.activation is not None:
            collect_attrs["activation"] = group.collect.activation
        if group.next_group is not None:
            collect_attrs["route_to"] = group.next_group

        collect_node = PlacedNode(
            id=collect_id,
            op=OpType.COLLECT,
            coord=(col, collect_row),
            data=PlacedCollectData(
                num_fragments=group.collect.num_fragments,
                total_rows=group.collect.total_rows,
                origin_id=group.origin_id,
                activation=group.collect.activation,
            ),
            attrs=collect_attrs,
        )
        placed_nodes.append(collect_node)

        # Internal edges: each tile → collect
        for i, tile_node in enumerate(tile_nodes):
            placed_edges.append(
                PlacedEdge(
                    src_node=tile_node.id,
                    src_slot=0,
                    dst_node=collect_id,
                    dst_slot=i,
                )
            )

        group_tile_nodes[group.origin_id] = tile_nodes
        group_collect_node[group.origin_id] = collect_node

        column_height = len(group.tiles) + 1
        if column_height > max_column_height:
            max_column_height = column_height
        col += 1

    # Explicit inter-group edges: collect → next group's tiles
    for group in expanded.groups:
        if group.next_group is not None:
            collect = group_collect_node[group.origin_id]
            next_tiles = group_tile_nodes[group.next_group]
            for i, tile in enumerate(next_tiles):
                placed_edges.append(
                    PlacedEdge(
                        src_node=collect.id,
                        src_slot=i,
                        dst_node=tile.id,
                        dst_slot=0,
                    )
                )

    width = col if col > 0 else 1
    height = max_column_height if max_column_height > 0 else 1
    if config.mesh_height is not None and height < config.mesh_height:
        height = config.mesh_height

    return SpatialIR(width=width, height=height, nodes=placed_nodes, edges=placed_edges)
```

- [ ] **Step 6: Wire expand into compiler pipeline**

In `python/meshflow/compiler/__init__.py`, update imports and pipeline:

```python
from meshflow.compiler.passes import expand, lower, place, route
```

And change the pipeline:
```python
# Before:
spatial = place(graph, config)

# After:
expanded = expand(graph, config)
spatial = place(expanded, config)
```

- [ ] **Step 7: Run all Python tests**

Run: `uv run pytest tests/python -v`
Expected: ALL tests pass — routing still works via `attrs` backward compat, placement tests verify typed `data`, compile/end-to-end tests pass through the full pipeline.

- [ ] **Step 8: Commit**

```bash
git add python/meshflow/compiler/spatial_ir.py python/meshflow/compiler/passes/place.py \
    python/meshflow/compiler/__init__.py tests/python/compiler/passes/test_place.py \
    tests/python/compiler/passes/test_route.py tests/python/compiler/test_compile.py
git commit -m "refactor: placement consumes ExpandedIR, produces typed PlacedNodeData

place() now takes ExpandedIR instead of GraphIR. PlacedNode carries
typed PlacedTileData/PlacedCollectData alongside attrs (temporary
backward compat for routing). Inter-group edges (collect → next tiles)
are now explicit in SpatialIR. Compiler pipeline calls expand() before
place()."
```

---

## Chunk 2: Routing Refactor + Cleanup

### Task 4: Refactor Routing to Use Typed Data + Explicit Edges

**Files:**
- Modify: `python/meshflow/compiler/passes/route.py`
- Modify: `tests/python/compiler/passes/test_route.py`

The routing pass currently reverse-engineers structure from attrs (`tiles_by_origin`, `route_to`). After this task, it reads typed `PlacedNodeData` and uses explicit SpatialIR edges for inter-group routing.

- [ ] **Step 1: Rewrite routing pass**

```python
# python/meshflow/compiler/passes/route.py
"""Routing pass — generates concrete hop lists and flattens to per-PE schedules."""

import numpy as np

from meshflow.compiler.config import CompilerConfig, RoutingStrategy
from meshflow.compiler.graph_ir import OpType
from meshflow.compiler.schedule_ir import (
    CollectOutputEntry,
    ConcatCollectEntry,
    ConcatCollectForwardEntry,
    Direction,
    ForwardActivationEntry,
    InputSlot,
    LinearEntry,
    PESchedule,
    ScheduleIR,
    TaskEntry,
)
from meshflow.compiler.spatial_ir import PlacedCollectData, PlacedTileData, SpatialIR


def route(
    spatial: SpatialIR,
    config: CompilerConfig,
    weights: dict[str, dict[str, np.ndarray]] | None = None,
) -> ScheduleIR:
    """Generate routes for all edges and flatten to per-PE task schedules."""
    if config.routing == RoutingStrategy.DIMENSION_ORDERED_XY:
        return _route_xy(spatial, weights)

    raise ValueError(f"unknown routing strategy: {config.routing!r}")


def _route_xy(
    spatial: SpatialIR,
    weights: dict[str, dict[str, np.ndarray]] | None = None,
) -> ScheduleIR:
    """Dimension-ordered XY routing: X first, then Y."""
    node_map = {n.id: n for n in spatial.nodes}

    pe_tasks: dict[tuple[int, int], list[TaskEntry]] = {}
    pe_sram: dict[tuple[int, int], dict[int, list[float]]] = {}
    for node in spatial.nodes:
        pe_tasks.setdefault(node.coord, [])
        pe_sram.setdefault(node.coord, {})

    for node in spatial.nodes:
        if node.op == OpType.FORWARD:
            outgoing = [e for e in spatial.edges if e.src_node == node.id]
            if not outgoing:
                continue
            dst_node = node_map[outgoing[0].dst_node]
            hops = _generate_route_xy(node.coord, dst_node.coord)
            pe_tasks[node.coord].append(
                ForwardActivationEntry(
                    trigger_slot=0,
                    input_slot=0,
                    route_dest=dst_node.coord,
                    route_hops=hops,
                )
            )

        elif isinstance(node.data, PlacedTileData):
            data = node.data
            # Find outgoing edge to collect PE
            outgoing = [e for e in spatial.edges if e.src_node == node.id]
            if not outgoing:
                raise ValueError(f"LINEAR tile node {node.id!r} has no outgoing edge")
            collect_node = node_map[outgoing[0].dst_node]
            hops = _generate_route_xy(node.coord, collect_node.coord)

            pe_tasks[node.coord].append(
                LinearEntry(
                    trigger_slot=0,
                    input_slot=0,
                    weight_slot=1,
                    bias_slot=2,
                    tile_rows=data.tile_rows,
                    tile_cols=data.in_features,
                    route_dest=collect_node.coord,
                    route_hops=hops,
                    fragment_slot=data.tile_index,
                    fragment_offset=data.fragment_offset,
                )
            )

            # Weight/bias SRAM
            if weights is not None and data.origin_id in weights:
                w = weights[data.origin_id]["weight"]
                b = weights[data.origin_id]["bias"]
                weight_tile = (
                    w[data.fragment_offset : data.fragment_offset + data.tile_rows, :]
                    .flatten()
                    .tolist()
                )
                bias_tile = (
                    b[data.fragment_offset : data.fragment_offset + data.tile_rows]
                    .flatten()
                    .tolist()
                )
                pe_sram[node.coord][1] = weight_tile
                pe_sram[node.coord][2] = bias_tile

        elif isinstance(node.data, PlacedCollectData):
            data = node.data

            # Determine intermediate vs terminal from explicit outgoing edges
            outgoing = [e for e in spatial.edges if e.src_node == node.id]
            is_intermediate = len(outgoing) > 0

            base = data.total_rows // data.num_fragments
            remainder = data.total_rows % data.num_fragments

            if is_intermediate:
                # Build route_dests from explicit edges
                route_dests: list[tuple[tuple[int, int], list[Direction]]] = []
                for edge in outgoing:
                    dst = node_map[edge.dst_node]
                    tile_hops = _generate_route_xy(node.coord, dst.coord)
                    route_dests.append((dst.coord, tile_hops))

                for i in range(data.num_fragments):
                    frag_offset = i * base + min(i, remainder)
                    pe_tasks[node.coord].append(
                        ConcatCollectForwardEntry(
                            trigger_slot=i,
                            num_fragments=data.num_fragments,
                            total_rows=data.total_rows,
                            fragment_offset=frag_offset,
                            activation=data.activation,
                            route_dests=list(route_dests),
                        )
                    )
            else:
                # Terminal layer
                for i in range(data.num_fragments):
                    frag_offset = i * base + min(i, remainder)
                    pe_tasks[node.coord].append(
                        ConcatCollectEntry(
                            trigger_slot=i,
                            num_fragments=data.num_fragments,
                            total_rows=data.total_rows,
                            fragment_offset=frag_offset,
                        )
                    )

        elif node.op == OpType.COLLECT:
            # Simple collect (M2 style — no typed data)
            pe_tasks[node.coord].append(
                CollectOutputEntry(trigger_slot=0, input_slot=0)
            )

    # Input slots
    first_layer_origin = _find_first_layer_origin(spatial)
    input_slots: list[InputSlot] = []
    for n in spatial.nodes:
        if isinstance(n.data, PlacedTileData):
            if first_layer_origin is not None and n.data.origin_id == first_layer_origin:
                input_slots.append(
                    InputSlot(name=n.data.origin_id, coord=n.coord, payload_slot=0)
                )
        elif isinstance(n.data, PlacedCollectData):
            continue
        elif n.id not in {e.dst_node for e in spatial.edges}:
            input_slots.append(InputSlot(name=n.id, coord=n.coord, payload_slot=0))

    pe_schedules = [
        PESchedule(coord=coord, tasks=tasks, initial_sram=pe_sram.get(coord, {}))
        for coord, tasks in pe_tasks.items()
    ]

    return ScheduleIR(
        width=spatial.width,
        height=spatial.height,
        pe_schedules=pe_schedules,
        input_slots=input_slots,
    )


def _find_first_layer_origin(spatial: SpatialIR) -> str | None:
    """Find the origin_id of the first LINEAR layer (leftmost column, x=0)."""
    for node in spatial.nodes:
        if isinstance(node.data, PlacedTileData) and node.coord[0] == 0:
            return node.data.origin_id
    return None


def _generate_route_xy(
    src: tuple[int, int], dst: tuple[int, int]
) -> list[Direction]:
    """Generate XY dimension-ordered hop list: X first, then Y."""
    hops: list[Direction] = []
    sx, sy = src
    dx, dy = dst

    while sx != dx:
        if dx > sx:
            hops.append(Direction.EAST)
            sx += 1
        else:
            hops.append(Direction.WEST)
            sx -= 1

    while sy != dy:
        if dy > sy:
            hops.append(Direction.NORTH)
            sy += 1
        else:
            hops.append(Direction.SOUTH)
            sy -= 1

    return hops
```

- [ ] **Step 2: Update routing tests**

In `tests/python/compiler/passes/test_route.py`, the routing tests that construct SpatialIR manually need to use typed data and explicit inter-group edges. Tests that go through `place()` were already updated in Task 3 to call `expand()` first.

For tests that construct SpatialIR directly:

1. Add imports: `from meshflow.compiler.spatial_ir import PlacedCollectData, PlacedTileData`
2. Replace `attrs={...}` with typed `data=PlacedTileData(...)` or `data=PlacedCollectData(...)` on PlacedNodes. Remove `attrs=` keyword.
3. For FORWARD/COLLECT nodes (M2 style), use `data=None` (default — no change needed)
4. For multi-layer tests, add explicit collect→tile edges and remove `route_to` from collect attrs
5. Remove any remaining `attrs=` keyword args

Example transformation for a LINEAR tile node:
```python
# Before:
PlacedNode(
    id="lin_tile_0", op=OpType.LINEAR, coord=(0, 0),
    attrs={"tile_index": 0, "tile_rows": 2, "fragment_offset": 0,
           "in_features": 4, "origin_id": "lin"},
)

# After:
PlacedNode(
    id="lin_tile_0", op=OpType.LINEAR, coord=(0, 0),
    data=PlacedTileData(
        tile_index=0, tile_rows=2, fragment_offset=0,
        in_features=4, origin_id="lin",
    ),
)
```

Example transformation for a COLLECT node:
```python
# Before:
PlacedNode(
    id="lin_collect", op=OpType.COLLECT, coord=(0, 2),
    attrs={"num_fragments": 2, "total_rows": 4, "origin_id": "lin"},
)

# After:
PlacedNode(
    id="lin_collect", op=OpType.COLLECT, coord=(0, 2),
    data=PlacedCollectData(
        num_fragments=2, total_rows=4, origin_id="lin",
    ),
)
```

For multi-layer tests, add explicit inter-group edges:
```python
# Before (edges only had tile→collect, collect had route_to attr):
edges = [
    PlacedEdge(src_node="l1_tile_0", src_slot=0, dst_node="l1_collect", dst_slot=0),
    PlacedEdge(src_node="l2_tile_0", src_slot=0, dst_node="l2_collect", dst_slot=0),
]

# After (also includes collect→tile inter-group edges):
edges = [
    PlacedEdge(src_node="l1_tile_0", src_slot=0, dst_node="l1_collect", dst_slot=0),
    PlacedEdge(src_node="l1_collect", src_slot=0, dst_node="l2_tile_0", dst_slot=0),
    PlacedEdge(src_node="l2_tile_0", src_slot=0, dst_node="l2_collect", dst_slot=0),
]
```

- [ ] **Step 3: Run all Python tests**

Run: `uv run pytest tests/python -v`
Expected: All tests pass

- [ ] **Step 4: Run Rust tests to verify end-to-end**

Run: `cargo test -p mesh_runtime`
Expected: All Rust tests pass (artifact format unchanged)

- [ ] **Step 5: Run linters**

Run: `uv run ruff check python tests && uv run ruff format --check python tests && uv run mypy python/meshflow`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add python/meshflow/compiler/passes/route.py tests/python/compiler/passes/test_route.py
git commit -m "refactor: routing uses typed PlacedNodeData and explicit edges

route.py reads PlacedTileData/PlacedCollectData instead of untyped
attrs dicts. Inter-group routing uses explicit collect→tile edges
from SpatialIR instead of route_to attr + tiles_by_origin index."
```

---

### Task 5: Remove `attrs` + Final Cleanup

**Files:**
- Modify: `python/meshflow/compiler/spatial_ir.py` — remove `attrs` field from PlacedNode
- Modify: `python/meshflow/compiler/passes/place.py` — remove attrs production
- Modify: `tests/python/compiler/passes/test_place.py` — remove any remaining attrs assertions
- Modify: `tests/python/compiler/passes/test_route.py` — remove any remaining attrs references
- Verify: all other files

- [ ] **Step 1: Remove `attrs` field from PlacedNode**

In `python/meshflow/compiler/spatial_ir.py`, remove the `attrs` field and the `Any` import:

```python
# Remove this line:
from typing import Any

# Remove this field from PlacedNode:
    attrs: dict[str, Any] | None = None
```

- [ ] **Step 2: Remove attrs production from placement**

In `python/meshflow/compiler/passes/place.py`, remove the `tile_attrs = {...}` dict construction and the `attrs=tile_attrs` / `attrs=collect_attrs` keyword args from PlacedNode construction in `_place_linear_columns`.

- [ ] **Step 3: Search for any remaining `attrs` references**

Search compiler code and tests for `.attrs` references. Note: `node.attrs` in `python/meshflow/compiler/__init__.py` (the `_validate_weights` and `_validate_shape_chaining` functions) references `GraphIR` Node attrs, NOT PlacedNode attrs — those are correct and should NOT be changed.

- [ ] **Step 4: Remove unused imports**

Check all modified files for unused imports (e.g., `Any` from `spatial_ir.py`).

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/python -v && cargo test -p mesh_runtime`
Expected: All tests pass

- [ ] **Step 6: Run all linters**

Run:
```bash
uv run ruff check python tests
uv run ruff format --check python tests
uv run mypy python/meshflow
cargo fmt --check
cargo clippy -p mesh_runtime -- -D warnings
```
Expected: All clean

- [ ] **Step 7: Commit**

```bash
git add -u
git commit -m "refactor: remove deprecated attrs field from PlacedNode

Routing now uses typed PlacedNodeData exclusively. The temporary
attrs backward-compat field is no longer needed."
```
