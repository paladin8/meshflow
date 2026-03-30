"""Tests for the placement pass."""

import pytest
from meshflow.compiler import CompilerConfig
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.compiler.passes.expand import expand
from meshflow.compiler.passes.place import place
from meshflow.compiler.spatial_ir import (
    PlacedAttentionPeData,
    PlacedCollectData,
    PlacedNodeKind,
    PlacedRmsNormReduceData,
    PlacedRmsNormTileData,
    PlacedTileData,
)


class TestPlacement:
    def test_sequential_three_nodes(self) -> None:
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
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())

        assert spatial.width == 3
        assert spatial.height == 1
        coords = {n.id: n.coord for n in spatial.nodes}
        assert coords == {"a": (0, 0), "b": (1, 0), "c": (2, 0)}
        # Passthrough nodes have no typed data
        for n in spatial.nodes:
            assert n.data is None

    def test_auto_sizing_nx1(self) -> None:
        graph = GraphIR(
            nodes=[Node(id=f"n{i}", op=OpType.FORWARD) for i in range(5)],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())
        assert spatial.width == 5
        assert spatial.height == 1

    def test_explicit_mesh_dimensions(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="c", op=OpType.FORWARD),
                Node(id="d", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="d", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="d", dst_slot=0),
                Edge(src_node="c", src_slot=0, dst_node="d", dst_slot=0),
            ],
        )
        config = CompilerConfig(mesh_width=2, mesh_height=2)
        expanded = expand(graph, config)
        spatial = place(expanded, config)

        assert spatial.width == 2
        assert spatial.height == 2
        coords = {n.id: n.coord for n in spatial.nodes}
        assert coords["a"] == (0, 0)
        assert coords["b"] == (1, 0)
        assert coords["c"] == (0, 1)
        assert coords["d"] == (1, 1)

    def test_too_many_nodes_for_mesh(self) -> None:
        graph = GraphIR(
            nodes=[Node(id=f"n{i}", op=OpType.FORWARD) for i in range(5)],
            edges=[],
        )
        config = CompilerConfig(mesh_width=2, mesh_height=2)
        expanded = expand(graph, config)
        with pytest.raises(ValueError, match="does not fit"):
            place(expanded, config)

    def test_single_node(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="x", op=OpType.COLLECT)],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())
        assert spatial.width == 1
        assert spatial.height == 1
        assert spatial.nodes[0].coord == (0, 0)

    def test_empty_graph(self) -> None:
        graph = GraphIR(nodes=[], edges=[])
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())
        assert spatial.width == 1
        assert spatial.height == 1
        assert spatial.nodes == []

    def test_edges_preserved(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
        )
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())
        assert len(spatial.edges) == 1
        assert spatial.edges[0].src_node == "a"
        assert spatial.edges[0].dst_node == "b"

    def test_validates_graph(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="a", op=OpType.FORWARD)],
            edges=[Edge(src_node="a", src_slot=0, dst_node="missing", dst_slot=0)],
        )
        with pytest.raises(ValueError, match="unknown destination node"):
            expand(graph, CompilerConfig())


class TestLinearPlacement:
    def _make_linear_graph(self, in_f: int = 4, out_f: int = 6) -> GraphIR:
        return GraphIR(
            nodes=[
                Node(
                    id="linear1",
                    op=OpType.LINEAR,
                    attrs={"in_features": in_f, "out_features": out_f},
                )
            ],
            edges=[],
        )

    def test_linear_expands_to_tiles_and_collect(self) -> None:
        graph = self._make_linear_graph(in_f=4, out_f=6)
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)

        # 3 tile PEs + 1 collect = 4 placed nodes, vertical layout
        assert len(spatial.nodes) == 4
        assert spatial.width == 1
        assert spatial.height == 4

        # 3 tile PEs with collect in the middle (row 1)
        # Tiles: row 0, row 2, row 3  |  Collect: row 1
        tile_node_map = {n.id: n for n in spatial.nodes if n.kind == PlacedNodeKind.LINEAR_TILE}
        for i in range(3):
            tile = tile_node_map[f"linear1_tile_{i}"]
            assert tile.kind == PlacedNodeKind.LINEAR_TILE
            assert isinstance(tile.data, PlacedTileData)
            assert tile.data.tile_index == i
            assert tile.data.tile_rows == 2
            assert tile.data.in_features == 4
            assert tile.data.origin_id == "linear1"

        # Collect PE is near the center of the column (staggered by col % 3)
        collect = next(n for n in spatial.nodes if n.id == "linear1_collect")
        assert collect.kind == PlacedNodeKind.LINEAR_COLLECT
        # Collect should be within the column and not on the same row as any tile
        tile_ys = [tile_node_map[f"linear1_tile_{i}"].coord[1] for i in range(3)]
        assert collect.coord[1] not in tile_ys
        assert 0 <= collect.coord[1] <= max(tile_ys)
        assert isinstance(collect.data, PlacedCollectData)
        assert collect.data.num_fragments == 3
        assert collect.data.total_rows == 6

    def test_linear_generates_internal_edges(self) -> None:
        graph = self._make_linear_graph(in_f=4, out_f=6)
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)

        # 3 internal edges: tile_0->collect, tile_1->collect, tile_2->collect
        internal_edges = [e for e in spatial.edges if e.dst_node == "linear1_collect"]
        assert len(internal_edges) == 3
        for i, edge in enumerate(internal_edges):
            assert edge.src_node == f"linear1_tile_{i}"
            assert edge.dst_node == "linear1_collect"
            assert edge.dst_slot == i

    def test_linear_auto_sizing(self) -> None:
        """Without explicit mesh_height, one PE per output row."""
        graph = self._make_linear_graph(in_f=2, out_f=3)
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())

        # 3 tiles + 1 collect = 4 nodes, height=4 (vertical)
        assert len(spatial.nodes) == 4
        assert spatial.width == 1
        assert spatial.height == 4

    def test_linear_validation_missing_attrs(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="bad", op=OpType.LINEAR)],
            edges=[],
        )
        with pytest.raises(ValueError, match="requires attrs"):
            expand(graph, CompilerConfig())

    def test_linear_uneven_tiling(self) -> None:
        """out_features=7 with 3 tiles uses base/remainder distribution."""
        graph = GraphIR(
            nodes=[
                Node(
                    id="linear1",
                    op=OpType.LINEAR,
                    attrs={"in_features": 4, "out_features": 7},
                )
            ],
            edges=[],
        )
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)

        # 3 tiles: base=2, remainder=1 → tiles get [3, 2, 2] rows
        assert len(spatial.nodes) == 4
        tile0 = spatial.nodes[0]
        tile1 = spatial.nodes[1]
        tile2 = spatial.nodes[2]
        assert isinstance(tile0.data, PlacedTileData)
        assert tile0.data.tile_rows == 3
        assert tile0.data.fragment_offset == 0
        assert isinstance(tile1.data, PlacedTileData)
        assert tile1.data.tile_rows == 2
        assert tile1.data.fragment_offset == 3
        assert isinstance(tile2.data, PlacedTileData)
        assert tile2.data.tile_rows == 2
        assert tile2.data.fragment_offset == 5


class TestMultiLayerPlacement:
    def test_two_layer_mlp_placement(self) -> None:
        """Linear(4,8) → ReLU → Linear(8,3) with mesh_height=4."""
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 8}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 8, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)

        # 2 columns: l1 (3 tiles + collect), l2 (3 tiles + collect)
        assert spatial.width == 2
        assert spatial.height == 4

        coords = {n.id: n.coord for n in spatial.nodes}
        # Column 0: l1 tiles with collect in middle
        assert coords["l1_tile_0"][0] == 0
        assert coords["l1_collect"][0] == 0
        # Collect is within the column (staggered by col % 3)
        tile_ys = [coords[f"l1_tile_{i}"][1] for i in range(3)]
        assert coords["l1_collect"][1] not in tile_ys
        assert 0 <= coords["l1_collect"][1] <= max(tile_ys)
        # Column 1: l2 tiles with collect in middle
        assert coords["l2_tile_0"][0] == 1
        assert coords["l2_collect"][0] == 1

        # RELU node is NOT placed (fused onto collect)
        placed_ids = {n.id for n in spatial.nodes}
        assert "r1" not in placed_ids

        # l1 collect has activation via typed data
        l1_collect = next(n for n in spatial.nodes if n.id == "l1_collect")
        assert isinstance(l1_collect.data, PlacedCollectData)
        assert l1_collect.data.activation == "relu"

        # l2 collect is terminal (no activation)
        l2_collect = next(n for n in spatial.nodes if n.id == "l2_collect")
        assert isinstance(l2_collect.data, PlacedCollectData)
        assert l2_collect.data.activation is None

    def test_relu_fused_no_placed_node(self) -> None:
        """RELU nodes don't produce placed nodes."""
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        expanded = expand(graph, CompilerConfig(mesh_height=4))
        spatial = place(expanded, CompilerConfig(mesh_height=4))
        # Only LINEAR tiles + collect PEs, no RELU PE
        assert all(
            n.kind in (PlacedNodeKind.LINEAR_TILE, PlacedNodeKind.LINEAR_COLLECT)
            for n in spatial.nodes
        )

    def test_three_layer_mlp_placement(self) -> None:
        """Linear → ReLU → Linear → ReLU → Linear."""
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 8}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 8, "out_features": 6}),
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
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)

        assert spatial.width == 3
        assert spatial.height == 4

        # l1 collect has activation
        l1_collect = next(n for n in spatial.nodes if n.id == "l1_collect")
        assert isinstance(l1_collect.data, PlacedCollectData)
        assert l1_collect.data.activation == "relu"

        # l2 collect has activation
        l2_collect = next(n for n in spatial.nodes if n.id == "l2_collect")
        assert isinstance(l2_collect.data, PlacedCollectData)
        assert l2_collect.data.activation == "relu"

        # l3 collect is terminal
        l3_collect = next(n for n in spatial.nodes if n.id == "l3_collect")
        assert isinstance(l3_collect.data, PlacedCollectData)
        assert l3_collect.data.activation is None

    def test_linear_without_relu(self) -> None:
        """Direct Linear → Linear (no activation)."""
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6}),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[Edge(src_node="l1", src_slot=0, dst_node="l2", dst_slot=0)],
        )
        expanded = expand(graph, CompilerConfig(mesh_height=4))
        spatial = place(expanded, CompilerConfig(mesh_height=4))

        l1_collect = next(n for n in spatial.nodes if n.id == "l1_collect")
        assert isinstance(l1_collect.data, PlacedCollectData)
        assert l1_collect.data.activation is None

    def test_uneven_tiling_different_layers(self) -> None:
        """Layers with different out_features get different tile counts."""
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 7}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 7, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        spatial = place(expanded, config)

        # l1: 7 features, 3 tiles → [3, 2, 2] rows
        l1_tiles = [n for n in spatial.nodes if n.id.startswith("l1_tile")]
        assert len(l1_tiles) == 3
        assert isinstance(l1_tiles[0].data, PlacedTileData)
        assert l1_tiles[0].data.tile_rows == 3
        assert isinstance(l1_tiles[1].data, PlacedTileData)
        assert l1_tiles[1].data.tile_rows == 2
        assert isinstance(l1_tiles[2].data, PlacedTileData)
        assert l1_tiles[2].data.tile_rows == 2

        # l2: 3 features, 3 tiles → [1, 1, 1] rows
        l2_tiles = [n for n in spatial.nodes if n.id.startswith("l2_tile")]
        assert len(l2_tiles) == 3
        assert all(isinstance(t.data, PlacedTileData) and t.data.tile_rows == 1 for t in l2_tiles)

    def test_terminal_relu_placement(self) -> None:
        """LINEAR → RELU at end of graph: collect gets activation but no next group."""
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6}),
                Node(id="r1", op=OpType.RELU),
            ],
            edges=[Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0)],
        )
        expanded = expand(graph, CompilerConfig(mesh_height=4))
        spatial = place(expanded, CompilerConfig(mesh_height=4))
        l1_collect = next(n for n in spatial.nodes if n.id == "l1_collect")
        assert isinstance(l1_collect.data, PlacedCollectData)
        assert l1_collect.data.activation == "relu"

    def test_inter_layer_edges_explicit(self) -> None:
        """Inter-layer edges are now explicit collect → next tiles."""
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        expanded = expand(graph, CompilerConfig(mesh_height=4))
        spatial = place(expanded, CompilerConfig(mesh_height=4))

        # Internal tile→collect edges exist
        internal_edges = [e for e in spatial.edges if e.dst_node.endswith("_collect")]
        assert len(internal_edges) == 6  # 3 for l1, 3 for l2

        # Explicit inter-group edges: l1_collect → l2 tiles
        inter_group_edges = [e for e in spatial.edges if e.src_node == "l1_collect"]
        assert len(inter_group_edges) == 3
        dest_ids = sorted(e.dst_node for e in inter_group_edges)
        assert dest_ids == ["l2_tile_0", "l2_tile_1", "l2_tile_2"]


class TestRmsNormPlacement:
    def test_rmsnorm_column_layout(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="rn", op=OpType.RMSNORM, attrs={"eps": 1e-6, "feature_count": 4})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())

        # 4 tile PEs + 1 reduce PE + 1 collect PE = 6 nodes
        assert len(spatial.nodes) == 6
        coords = {n.id: n.coord for n in spatial.nodes}
        # All in column 0
        for i in range(4):
            assert coords[f"rn_tile_{i}"][0] == 0
        assert coords["rn_reduce"][0] == 0
        assert coords["rn_collect"][0] == 0
        # Collect is within the column (staggered by col % 3)
        tile_ys = [coords[f"rn_tile_{i}"][1] for i in range(4)]
        assert coords["rn_collect"][1] not in tile_ys
        assert 0 <= coords["rn_collect"][1] <= max(tile_ys)
        # Reduce is above all tiles and collect
        assert coords["rn_reduce"][1] > max(tile_ys)

    def test_rmsnorm_data_types(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="rn", op=OpType.RMSNORM, attrs={"eps": 1e-6, "feature_count": 4})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())

        tile = next(n for n in spatial.nodes if n.id == "rn_tile_0")
        assert isinstance(tile.data, PlacedRmsNormTileData)
        assert tile.data.tile_index == 0
        assert tile.data.origin_id == "rn"

        reduce_pe = next(n for n in spatial.nodes if n.id == "rn_reduce")
        assert isinstance(reduce_pe.data, PlacedRmsNormReduceData)
        assert reduce_pe.data.num_tiles == 4
        assert reduce_pe.data.feature_count == 4
        assert reduce_pe.data.eps == 1e-6

    def test_rmsnorm_internal_edges(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="rn", op=OpType.RMSNORM, attrs={"eps": 1e-6, "feature_count": 3})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())

        # 3 internal edges: tile → reduce
        internal = [e for e in spatial.edges if e.dst_node == "rn_reduce"]
        assert len(internal) == 3
        for i, edge in enumerate(internal):
            assert edge.src_node == f"rn_tile_{i}"
            assert edge.dst_slot == i


class TestAttentionPlacement:
    def test_attention_column_layout(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="mm", op=OpType.MATMUL, attrs={"seq_len": 4})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())

        assert len(spatial.nodes) == 4
        coords = {n.id: n.coord for n in spatial.nodes}
        for i in range(4):
            assert coords[f"mm_attn_{i}"] == (0, i)

    def test_attention_data_type(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="qkt", op=OpType.MATMUL, attrs={"seq_len": 2}),
                Node(id="sm", op=OpType.SOFTMAX),
                Node(id="av", op=OpType.MATMUL, attrs={"seq_len": 2}),
            ],
            edges=[
                Edge(src_node="qkt", src_slot=0, dst_node="sm", dst_slot=0),
                Edge(src_node="sm", src_slot=0, dst_node="av", dst_slot=0),
            ],
        )
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())

        # 2 attention PEs + 1 collect PE = 3 nodes
        assert len(spatial.nodes) == 3
        pe0 = spatial.nodes[0]
        assert isinstance(pe0.data, PlacedAttentionPeData)
        assert pe0.data.softmax_id == "sm"
        assert pe0.data.av_matmul_id == "av"


class TestAddPlacement:
    def test_add_single_pe(self) -> None:
        """ADD is a single-PE passthrough — placed in its own column."""
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="add", op=OpType.ADD),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="add", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="add", dst_slot=1),
            ],
        )
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())

        add_node = next(n for n in spatial.nodes if n.id == "add")
        assert add_node.kind == PlacedNodeKind.ADD
        assert add_node.data is None  # passthrough, no typed data

    def test_add_inter_group_edges(self) -> None:
        """FORWARD → ADD: 1:1 edges (both single PE)."""
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="add", op=OpType.ADD),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="add", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="add", dst_slot=1),
            ],
        )
        expanded = expand(graph, CompilerConfig())
        spatial = place(expanded, CompilerConfig())

        # Both a→add and b→add are 1:1 edges
        add_edges = [e for e in spatial.edges if e.dst_node == "add"]
        assert len(add_edges) == 2
        src_nodes = {e.src_node for e in add_edges}
        assert src_nodes == {"a", "b"}


class TestCompactPlacement:
    """Tests for the _compact_columns pass that merges single-PE groups."""

    def test_compact_single_pe_merged(self) -> None:
        """Single-PE groups (ADD, COLLECT) merge into adjacent columns."""
        # FORWARD → LINEAR(4→4) → ADD → COLLECT
        # Groups: FORWARD(col0), LINEAR(col1), ADD(col2), COLLECT(col3)
        # ADD should merge into LINEAR's column, COLLECT into ADD's (now LINEAR's)
        graph = GraphIR(
            nodes=[
                Node(id="input", op=OpType.FORWARD),
                Node(id="linear", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4}),
                Node(id="add", op=OpType.ADD),
                Node(id="output", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="input", src_slot=0, dst_node="linear", dst_slot=0),
                Edge(src_node="input", src_slot=0, dst_node="add", dst_slot=1),
                Edge(src_node="linear", src_slot=0, dst_node="add", dst_slot=0),
                Edge(src_node="add", src_slot=0, dst_node="output", dst_slot=0),
            ],
        )
        config = CompilerConfig(mesh_height=6)
        expanded = expand(graph, config)
        spatial = place(expanded, config)

        # Without compaction: 4 columns (FORWARD, LINEAR, ADD, COLLECT)
        # With co-location + compaction: ADD co-locates with LINEAR collect,
        # COLLECT merges into the same column. Width should be reduced.
        assert spatial.width < 4, f"expected compact width < 4, got {spatial.width}"

        # ADD node shares coordinate with the LINEAR collect PE (co-location)
        add_node = next(n for n in spatial.nodes if n.id == "add")
        collect_node = next(n for n in spatial.nodes if n.id == "linear_collect")
        assert add_node.coord == collect_node.coord, "ADD should co-locate with collect PE"

    def test_compact_preserves_multi_pe_columns(self) -> None:
        """Tiled groups keep their own columns and are never merge candidates."""
        graph = GraphIR(
            nodes=[
                Node(id="input", op=OpType.FORWARD),
                Node(id="linear", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4}),
                Node(id="output", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="input", src_slot=0, dst_node="linear", dst_slot=0),
                Edge(src_node="linear", src_slot=0, dst_node="output", dst_slot=0),
            ],
        )
        config = CompilerConfig(mesh_height=6)
        expanded = expand(graph, config)
        spatial = place(expanded, config)

        # LINEAR group has multiple PEs (tiles + collect) — must keep its own column
        linear_nodes = [n for n in spatial.nodes if n.kind == PlacedNodeKind.LINEAR_TILE]
        linear_cols = {n.coord[0] for n in linear_nodes}
        assert len(linear_cols) == 1, "linear tiles should all be in one column"

        # That column should contain only LINEAR nodes (tiles + collect)
        linear_col = linear_cols.pop()
        col_nodes = [n for n in spatial.nodes if n.coord[0] == linear_col]
        for n in col_nodes:
            assert n.kind in (
                PlacedNodeKind.LINEAR_TILE,
                PlacedNodeKind.LINEAR_COLLECT,
                PlacedNodeKind.COLLECT_SIMPLE,  # output may merge into this column
            ), f"unexpected {n.kind} in linear column"

    def test_compact_no_unexpected_coordinate_conflicts(self) -> None:
        """No unexpected coordinate conflicts after compaction.

        Co-located ADD+collect pairs intentionally share coordinates.
        All other nodes must have unique coordinates.
        """
        from meshflow.models.transformer import transformer_block

        for sl, dm, df, mh in [(4, 8, 16, 6), (8, 16, 32, 8)]:
            graph = transformer_block(sl, dm, df)
            config = CompilerConfig(mesh_height=mh)
            expanded = expand(graph, config)
            spatial = place(expanded, config)

            # ADD nodes co-located with collect PEs are expected duplicates
            add_ids = {n.id for n in spatial.nodes if n.kind == PlacedNodeKind.ADD}
            non_add_coords = [n.coord for n in spatial.nodes if n.id not in add_ids]
            assert len(non_add_coords) == len(set(non_add_coords)), (
                f"unexpected coordinate conflict in ({sl},{dm},{df},mh={mh})"
            )


class TestCollectAddColocation:
    """Tests for Phase 4 LinearCollect+Add co-location."""

    def test_collect_add_colocation(self) -> None:
        """Co-located PE has both ConcatCollectForward and Add tasks."""
        from meshflow.models.transformer import transformer_block

        graph = transformer_block(4, 8, 16)
        config = CompilerConfig(mesh_height=6)
        expanded = expand(graph, config)
        spatial = place(expanded, config)

        # add1 and add2 should co-locate with their upstream collect PEs
        for add_id in ["add1", "add2"]:
            add_node = next(n for n in spatial.nodes if n.id == add_id)
            # Find the collect PE at the same coordinate
            collect_at_same = [
                n
                for n in spatial.nodes
                if n.coord == add_node.coord and n.kind == PlacedNodeKind.LINEAR_COLLECT
            ]
            assert len(collect_at_same) == 1, (
                f"{add_id} should share coordinate with exactly 1 collect PE"
            )

    def test_add_pe_eliminated(self) -> None:
        """No separate column for ADD PEs when co-located."""
        from meshflow.models.transformer import transformer_block

        graph = transformer_block(4, 8, 16)
        config = CompilerConfig(mesh_height=6)
        expanded = expand(graph, config)
        spatial = place(expanded, config)

        # ADD nodes should share a column with a collect PE, not have their own
        for add_id in ["add1", "add2"]:
            add_node = next(n for n in spatial.nodes if n.id == add_id)
            col_nodes = [n for n in spatial.nodes if n.coord[0] == add_node.coord[0]]
            # The column should have more than just the ADD (it should have tiles + collect)
            assert len(col_nodes) > 1, f"{add_id} should share column with collect PE"

    def test_colocation_numerical_correctness(self) -> None:
        """Transformer block outputs match reference with co-location enabled."""
        import torch

        from meshflow._mesh_runtime import run_program
        from meshflow.compiler import compile
        from meshflow.compiler.artifact import serialize
        from meshflow.models.reference import reference_transformer_block
        from meshflow.models.transformer import transformer_block, transformer_weights

        seq_len, d_model, d_ff = 4, 8, 16
        graph = transformer_block(seq_len, d_model, d_ff)
        weights = transformer_weights(d_model, d_ff, seed=42)
        config = CompilerConfig(mesh_height=6)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(99)
        x = torch.randn(seq_len, d_model)
        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        expected = reference_transformer_block(x, weights)
        assert len(result.outputs) == 1
        actual = torch.tensor(next(iter(result.outputs.values())))
        assert torch.allclose(actual.view(seq_len, d_model), expected, atol=1e-4), (
            f"max diff: {(actual.view(seq_len, d_model) - expected).abs().max()}"
        )


class TestMiddleCollectPlacement:
    """Tests for middle collect PE placement with 3-way stagger."""

    def test_middle_collect_default_is_center(self) -> None:
        """With stagger_offset=1 (col%3==1), collect is at center (baseline)."""
        from meshflow.compiler.passes.place import _middle_collect_rows

        for n in range(1, 8):
            tile_rows, collect_row = _middle_collect_rows(n, stagger_offset=1)
            assert collect_row == n // 2, f"n={n}: collect at {collect_row}, expected {n // 2}"
            assert collect_row not in tile_rows
            all_rows = sorted(tile_rows + [collect_row])
            assert all_rows == list(range(n + 1)), f"n={n}: rows not contiguous: {all_rows}"

    def test_middle_collect_stagger_three_way(self) -> None:
        """3-way stagger cycles through center-1, center, center+1."""
        from meshflow.compiler.passes.place import _middle_collect_rows

        for n in range(2, 8):
            center = n // 2
            rows_by_offset = []
            for offset in range(3):
                _, collect_row = _middle_collect_rows(n, stagger_offset=offset)
                rows_by_offset.append(collect_row)
            # offset 0 → center-1, offset 1 → center, offset 2 → center+1
            assert rows_by_offset[1] == center
            assert rows_by_offset[0] <= center
            assert rows_by_offset[2] >= center
            if n >= 3:
                assert len(set(rows_by_offset)) == 3, f"n={n}: expected 3 distinct positions"
            else:
                assert len(set(rows_by_offset)) >= 2, f"n={n}: stagger produced no variation"

    def test_middle_collect_contiguous_rows(self) -> None:
        """All rows are unique and contiguous 0..n for every stagger offset."""
        from meshflow.compiler.passes.place import _middle_collect_rows

        for n in range(1, 8):
            for offset in range(6):  # test several offsets
                tile_rows, collect_row = _middle_collect_rows(n, stagger_offset=offset)
                assert collect_row not in tile_rows, f"n={n}, offset={offset}: collect conflicts"
                all_rows = sorted(tile_rows + [collect_row])
                assert all_rows == list(range(n + 1)), (
                    f"n={n}, offset={offset}: rows not contiguous: {all_rows}"
                )

    def test_middle_collect_reduces_max_internal_hops(self) -> None:
        """Max tile-to-collect distance is at most ceil(N/2) + 1 with stagger."""
        import math

        from meshflow.compiler.passes.place import _middle_collect_rows

        for n in range(1, 8):
            for offset in range(3):
                tile_rows, collect_row = _middle_collect_rows(n, stagger_offset=offset)
                max_dist = max(abs(r - collect_row) for r in tile_rows)
                # Stagger adds at most 1 hop vs center placement
                expected_max = math.ceil(n / 2) + 1
                assert max_dist <= expected_max, (
                    f"n={n}, offset={offset}: max distance {max_dist} exceeds {expected_max}"
                )
