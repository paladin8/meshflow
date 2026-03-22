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
