"""Compiler configuration and strategy enums."""

from dataclasses import dataclass
from enum import Enum


class PlacementStrategy(Enum):
    SEQUENTIAL = "sequential"


class RoutingStrategy(Enum):
    DIMENSION_ORDERED_XY = "xy"


@dataclass
class CompilerConfig:
    placement: PlacementStrategy = PlacementStrategy.SEQUENTIAL
    routing: RoutingStrategy = RoutingStrategy.DIMENSION_ORDERED_XY
    mesh_width: int | None = None
    mesh_height: int | None = None
    sram_capacity_bytes: int = 65536  # 64 KB per PE
