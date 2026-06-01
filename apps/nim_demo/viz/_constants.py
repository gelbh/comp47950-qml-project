"""Shared colours and kernel heatmap size caps for the Nim demo plots."""

from __future__ import annotations

COLOR_CHOSEN = "#2a9d8f"
COLOR_NEUTRAL = "#6c757d"
COLOR_OPTIMAL = "#e76f51"
COLOR_AGREE = "#2a9d8f"
COLOR_DISAGREE = "#e76f51"
COLOR_HEAP = ["#264653", "#2a9d8f", "#e9c46a"]
COLOR_STONE_EMPTY = "#e9ecef"

# Above this size, skip per-cell HTML hovers (memory + build time).
KERNEL_HEATMAP_RICH_HOVER_MAX_N = 48
# Above this size, hide per-index tick labels on the axes.
KERNEL_HEATMAP_AXIS_TICKS_MAX_N = 56
