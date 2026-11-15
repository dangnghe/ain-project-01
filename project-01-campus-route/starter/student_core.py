"""AI core for Project 01 (Smart Campus Route Planner).

Implements Breadth-First Search, Greedy Best-First Search, and A* from
scratch, plus Euclidean and Manhattan heuristics. Every search accepts the
two dynamic constraints used by the interactive product:

* ``blocked_edges``  -- undirected ``[a, b]`` pairs that must not be used
  (simulated maintenance work).
* ``accessible_only`` -- restrict traversal to edges flagged ``accessible``.

Searches return ``(path, expanded_node_count)`` where ``path`` is a list of
node ids from ``start`` to ``goal`` inclusive, or ``None`` when no route
exists. Pass a dict as ``stats`` to collect extra measurement fields
(``expanded``, ``frontier_peak``, ``generated``).

No external graph-search solver is used: only the standard library.
"""
from __future__ import annotations

import heapq
import math
import sys
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import Edge, Graph  # noqa: E402

# (path of node ids from start to goal inclusive, or None) and the number of
# nodes expanded while searching.
PathResult = Tuple[Optional[List[str]], int]

Heuristic = Callable[[Graph, str, str], float]

BlockedEdges = Optional[Iterable[Iterable[str]]]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _normalize_edge_pair(edge_a: str, edge_b: str) -> Tuple[str, str]:
    return (edge_a, edge_b) if edge_a <= edge_b else (edge_b, edge_a)


def _blocked_set(blocked_edges: BlockedEdges) -> Set[Tuple[str, str]]:
    if not blocked_edges:
        return set()
    return {
        _normalize_edge_pair(a, b)
        for a, b in blocked_edges
        if a is not None and b is not None
    }


def _is_blocked(edge: Edge, blocked_edges: BlockedEdges) -> bool:
    if not blocked_edges:
        return False
    blocked = _blocked_set(blocked_edges)
    return _normalize_edge_pair(edge.source, edge.target) in blocked


def _neighbors_for_node(
    graph: Graph,
    node_id: str,
    *,
    blocked_edges: BlockedEdges = None,
    accessible_only: bool = False,
) -> List[Tuple[str, Edge]]:
    """Every usable ``(other_node, edge)`` pair leaving ``node_id``."""
    blocked = _blocked_set(blocked_edges)
    neighbors: List[Tuple[str, Edge]] = []
    for edge in graph.neighbors(node_id):
        if blocked and _normalize_edge_pair(edge.source, edge.target) in blocked:
            continue
        if accessible_only and not edge.accessible:
            continue
        other = edge.target if edge.source == node_id else edge.source
        neighbors.append((other, edge))
    return neighbors


def _reconstruct_path(
    parents: Dict[str, Optional[str]], start: str, goal: str
) -> Optional[List[str]]:
    if goal not in parents:
        return None
    path: List[str] = []
    current: Optional[str] = goal
    while current is not None:
        path.append(current)
        if current == start:
            break
        current = parents[current]
    if not path or path[-1] != start:
        return None
    path.reverse()
    return path


def _record(
    stats: Optional[Dict[str, object]],
    *,
    expanded: int,
    frontier_peak: int,
    generated: int,
) -> None:
    if stats is not None:
        stats.update(
            expanded=expanded, frontier_peak=frontier_peak, generated=generated
        )


def path_cost(
    graph: Graph,
    path: Optional[List[str]],
    edge_weight: str = "distance",
    blocked_edges: BlockedEdges = None,
    accessible_only: bool = False,
) -> Optional[float]:
    """Sum the cheapest usable edge between consecutive nodes of ``path``.

    Returns ``None`` when ``path`` is ``None`` or a hop has no usable edge.
    """
    if edge_weight not in {"distance", "travel_time"}:
        raise ValueError(f"Unsupported edge_weight: {edge_weight!r}")
    if not path:
        return None
    blocked = _blocked_set(blocked_edges)
    total = 0.0
    for current, nxt in zip(path, path[1:]):
        best: Optional[float] = None
        for other, edge in _neighbors_for_node(
            graph, current, accessible_only=accessible_only
        ):
            if other != nxt:
                continue
            if blocked and _normalize_edge_pair(edge.source, edge.target) in blocked:
                continue
            value = float(getattr(edge, edge_weight))
            if best is None or value < best:
                best = value
        if best is None:
            return None
        total += best
    return total


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def euclidean_heuristic(graph: Graph, node_id: str, goal_id: str) -> float:
    """Straight-line distance between two nodes in schematic units."""
    start_node = graph.get_node(node_id)
    goal_node = graph.get_node(goal_id)
    return math.hypot(start_node.x - goal_node.x, start_node.y - goal_node.y)


def manhattan_heuristic(graph: Graph, node_id: str, goal_id: str) -> float:
    """Manhattan (|dx| + |dy|) distance between two nodes."""
    start_node = graph.get_node(node_id)
    goal_node = graph.get_node(goal_id)
    return abs(start_node.x - goal_node.x) + abs(start_node.y - goal_node.y)


HEURISTICS: Dict[str, Heuristic] = {
    "euclidean": euclidean_heuristic,
    "manhattan": manhattan_heuristic,
}


def min_cost_per_unit(graph: Graph, edge_weight: str = "distance") -> float:
    """Smallest cost/geometric-distance ratio over all edges.

    Scaling a geometric heuristic by this factor keeps it admissible when the
    step cost is ``travel_time`` instead of ``distance`` (walking speed varies
    per corridor, so time can be smaller than the metric length).
    """
    if edge_weight == "distance":
        return 1.0
    if edge_weight != "travel_time":
        raise ValueError(f"Unsupported edge_weight: {edge_weight!r}")
    ratios: List[float] = []
    seen: Set[Tuple[str, str]] = set()
    for node_id in graph.node_ids:
        for edge in graph.neighbors(node_id):
            key = _normalize_edge_pair(edge.source, edge.target)
            if key in seen or edge.distance <= 0:
                continue
            seen.add(key)
            ratios.append(float(edge.travel_time) / float(edge.distance))
    return min(ratios) if ratios else 1.0


def make_heuristic(
    name: str = "euclidean", graph: Optional[Graph] = None, edge_weight: str = "distance"
) -> Heuristic:
    """Build an admissible heuristic for the chosen cost function.

    ``make_heuristic("euclidean")`` returns the plain geometric heuristic;
    with ``edge_weight="travel_time"`` the result is scaled by the minimum
    observed cost-per-metre so it never overestimates.
    """
    key = name.lower()
    if key not in HEURISTICS:
        raise ValueError(f"Unknown heuristic: {name!r}")
    base = HEURISTICS[key]
    if edge_weight == "distance" or graph is None:
        return base
    factor = min_cost_per_unit(graph, edge_weight)
    if factor == 1.0:
        return base

    def scaled(graph_: Graph, node_id: str, goal_id: str) -> float:
        return base(graph_, node_id, goal_id) * factor

    scaled.__name__ = f"{key}_scaled"
    return scaled


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------


def bfs(
    graph: Graph,
    start: str,
    goal: str,
    blocked_edges: BlockedEdges = None,
    accessible_only: bool = False,
    stats: Optional[Dict[str, object]] = None,
) -> PathResult:
    """Breadth-First Search: fewest hops, ignoring edge cost.

    ``path`` is a list of node ids from ``start`` to ``goal`` inclusive, or
    ``None`` if no route exists under the active constraints.
    """
    if start == goal:
        _record(stats, expanded=0, frontier_peak=0, generated=1)
        return [start], 0

    queue: Deque[str] = deque([start])
    parents: Dict[str, Optional[str]] = {start: None}
    visited: Set[str] = {start}
    expanded = 0
    frontier_peak = 1

    while queue:
        node = queue.popleft()
        expanded += 1

        if node == goal:
            _record(
                stats,
                expanded=expanded,
                frontier_peak=frontier_peak,
                generated=len(parents),
            )
            return _reconstruct_path(parents, start, goal), expanded

        for other, _edge in _neighbors_for_node(
            graph, node, blocked_edges=blocked_edges, accessible_only=accessible_only
        ):
            if other in visited:
                continue
            visited.add(other)
            parents[other] = node
            queue.append(other)
        frontier_peak = max(frontier_peak, len(queue))

    _record(stats, expanded=expanded, frontier_peak=frontier_peak, generated=len(parents))
    return None, expanded


def greedy(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Heuristic,
    blocked_edges: BlockedEdges = None,
    accessible_only: bool = False,
    stats: Optional[Dict[str, object]] = None,
) -> PathResult:
    """Greedy Best-First Search: order the frontier by ``heuristic`` only."""
    if start == goal:
        _record(stats, expanded=0, frontier_peak=0, generated=1)
        return [start], 0

    frontier: List[Tuple[float, int, str]] = [
        (heuristic(graph, start, goal), 0, start)
    ]
    parents: Dict[str, Optional[str]] = {start: None}
    visited: Set[str] = set()
    expanded = 0
    frontier_peak = 1
    tie = 0

    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        expanded += 1

        if node == goal:
            _record(
                stats,
                expanded=expanded,
                frontier_peak=frontier_peak,
                generated=len(parents),
            )
            return _reconstruct_path(parents, start, goal), expanded

        for other, _edge in _neighbors_for_node(
            graph, node, blocked_edges=blocked_edges, accessible_only=accessible_only
        ):
            if other in visited or other in parents:
                continue
            parents[other] = node
            tie += 1
            heapq.heappush(frontier, (heuristic(graph, other, goal), tie, other))
        frontier_peak = max(frontier_peak, len(frontier))

    _record(stats, expanded=expanded, frontier_peak=frontier_peak, generated=len(parents))
    return None, expanded


def astar(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Heuristic,
    edge_weight: str = "distance",
    blocked_edges: BlockedEdges = None,
    accessible_only: bool = False,
    stats: Optional[Dict[str, object]] = None,
) -> PathResult:
    """A* search with ``f = g + h``.

    ``edge_weight`` names the step-cost attribute (``"distance"`` or
    ``"travel_time"``). A closed set guarantees each node is expanded once,
    which is optimal for a consistent heuristic.
    """
    if edge_weight not in {"distance", "travel_time"}:
        raise ValueError(f"Unsupported edge_weight: {edge_weight!r}")
    if start == goal:
        _record(stats, expanded=0, frontier_peak=0, generated=1)
        return [start], 0

    tie = 0
    frontier: List[Tuple[float, int, str]] = [
        (heuristic(graph, start, goal), tie, start)
    ]
    g_score: Dict[str, float] = {start: 0.0}
    parents: Dict[str, Optional[str]] = {start: None}
    closed: Set[str] = set()
    expanded = 0
    frontier_peak = 1

    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1

        if node == goal:
            _record(
                stats,
                expanded=expanded,
                frontier_peak=frontier_peak,
                generated=len(parents),
            )
            return _reconstruct_path(parents, start, goal), expanded

        for other, edge in _neighbors_for_node(
            graph, node, blocked_edges=blocked_edges, accessible_only=accessible_only
        ):
            if other in closed:
                continue
            tentative = g_score[node] + float(getattr(edge, edge_weight))
            if tentative < g_score.get(other, math.inf):
                g_score[other] = tentative
                parents[other] = node
                tie += 1
                heapq.heappush(
                    frontier,
                    (tentative + heuristic(graph, other, goal), tie, other),
                )
        frontier_peak = max(frontier_peak, len(frontier))

    _record(stats, expanded=expanded, frontier_peak=frontier_peak, generated=len(g_score))
    return None, expanded
