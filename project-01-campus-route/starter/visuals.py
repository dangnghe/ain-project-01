"""Matplotlib rendering of the campus graph, constraints, and routes.

Used by the Streamlit product app to draw the schematic map with the blocked
corridors in red and every algorithm's route overlaid in its own colour.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D  

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import Graph  

BACKGROUND = "#fbfbfa"
EDGE_COLOR = "#b9bec7"
STAIRS_COLOR = "#c9871b"
BLOCKED_COLOR = "#e03131"
START_COLOR = "#1c7ed6"
GOAL_COLOR = "#d6336c"

NODE_COLORS = {
    "plaza": "#8d99ae",
    "entrance": "#4dabf7",
    "hallway": "#adb5bd",
    "room": "#868e96",
}


def _unique_edges(graph: Graph) -> List[Tuple[str, str, bool]]:
    """One ``(a, b, accessible)`` triple per undirected node pair."""
    seen: Dict[Tuple[str, str], bool] = {}
    for node_id in graph.node_ids:
        for edge in graph.neighbors(node_id):
            key = tuple(sorted((edge.source, edge.target)))
            seen[key] = seen.get(key, False) or edge.accessible
    return [(a, b, acc) for (a, b), acc in seen.items()]


def render_map(
    graph: Graph,
    routes: Optional[Dict[str, Tuple[Optional[Sequence[str]], str]]] = None,
    blocked_edges: Optional[Iterable[Iterable[str]]] = None,
    accessible_only: bool = False,
    start: Optional[str] = None,
    goal: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 7.6),
):
    """Draw the campus and (optionally) one route per algorithm.

    ``routes`` maps a label to ``(path, colour)``; ``path=None`` means the
    algorithm found no route and is simply not drawn.
    """
    blocked = {tuple(sorted((a, b))) for a, b in (blocked_edges or [])}
    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    for a, b, accessible in _unique_edges(graph):
        node_a, node_b = graph.get_node(a), graph.get_node(b)
        is_blocked = (a, b) in blocked
        hidden = accessible_only and not accessible and not is_blocked
        if is_blocked:
            ax.plot(
                [node_a.x, node_b.x],
                [node_a.y, node_b.y],
                color=BLOCKED_COLOR,
                lw=4.0,
                ls=(0, (3, 3)),
                alpha=0.95,
                zorder=2,
                solid_capstyle="round",
            )
            continue
        ax.plot(
            [node_a.x, node_b.x],
            [node_a.y, node_b.y],
            color=EDGE_COLOR if accessible else STAIRS_COLOR,
            lw=1.4 if accessible else 1.2,
            ls="-" if accessible else (0, (4, 2)),
            alpha=0.35 if hidden else 0.85,
            zorder=1,
        )

    for node_id in graph.node_ids:
        node = graph.get_node(node_id)
        alpha = 0.4 if accessible_only and node.node_type == "room" else 2
        ax.scatter(
            node.x,
            node.y,
            s=26,
            color=NODE_COLORS.get(node.node_type, "#868e96"),
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            alpha=0.85 if alpha == 0.4 else 0.95,
        )

    colors_used: List[Tuple[str, str]] = []
    if routes:
        for label, (path, color) in routes.items():
            if not path or len(path) < 2:
                continue
            xs = [graph.get_node(n).x for n in path]
            ys = [graph.get_node(n).y for n in path]
            ax.plot(
                xs,
                ys,
                color=color,
                lw=3.4,
                alpha=0.9,
                zorder=4,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            ax.scatter(xs, ys, s=16, color=color, zorder=5, edgecolors="white", linewidths=0.7)
            colors_used.append((label, color))

    for node_id, color, marker in ((start, START_COLOR, "*"), (goal, GOAL_COLOR, "X")):
        if node_id is None:
            continue
        node = graph.get_node(node_id)
        ax.scatter(
            node.x,
            node.y,
            s=340 if marker == "*" else 200,
            color=color,
            marker=marker,
            zorder=6,
            edgecolors="white",
            linewidths=1.1,
        )

    labels = []
    if start:
        labels.append((start, START_COLOR))
    if goal:
        labels.append((goal, GOAL_COLOR))
    for node_id, color in labels:
        node = graph.get_node(node_id)
        ax.annotate(
            node.name,
            (node.x, node.y),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=0.8, alpha=0.92),
            zorder=7,
        )

    ax.set_aspect("equal")
    ax.invert_yaxis() 
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    legend_handles = [
        Line2D([0], [0], color=EDGE_COLOR, lw=2, label="Walkable"),
        Line2D([0], [0], color=STAIRS_COLOR, lw=2, ls=(0, (4, 2)), label="Stairs only"),
        Line2D([0], [0], color=BLOCKED_COLOR, lw=3, ls=(0, (3, 3)), label="Blocked (maintenance)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=START_COLOR, markersize=14, label="Start"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=GOAL_COLOR, markersize=11, label="Goal"),
    ]
    legend_handles += [
        Line2D([0], [0], color=color, lw=3, label=label) for label, color in colors_used
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#dee2e6",
        bbox_to_anchor=(0.0, 1.0),
    )
    fig.tight_layout()
    return fig


def render_benchmark_chart(frame, metric: str = "expanded"):
    """Grouped bar chart of a benchmark metric per scenario and algorithm."""
    import pandas as pd

    pivot = frame.pivot_table(index="Scenario", columns="Algorithm", values=metric, aggfunc="min")
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=140)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    pivot.plot(kind="bar", ax=ax, width=0.8, colormap="viridis", edgecolor="none")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("")
    ax.grid(axis="y", color="#dee2e6", lw=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=len(pivot.columns), loc="upper center", bbox_to_anchor=(0.5, 1.14))
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8.5)
    fig.tight_layout()
    return fig
