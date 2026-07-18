"""Generic pixel-space region geometry and layouts."""

from .geometry import CanvasGeometry, PixelBox, align_up
from .layout import (
    REGION_ROLES,
    RegionDefinition,
    SpatialLayout,
    compile_spatial_layout,
)

__all__ = [
    "CanvasGeometry",
    "PixelBox",
    "REGION_ROLES",
    "RegionDefinition",
    "SpatialLayout",
    "align_up",
    "compile_spatial_layout",
]
