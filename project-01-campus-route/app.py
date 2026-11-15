from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))

from graph import load_graph, load_scenarios
import student_core
import visuals

ALGORITHMS = ("BFS", "Greedy", "A*")
COLORS = {"BFS": "#7048e8", "Greedy": "#f08c00", "A*": "#2f9e44"}
ALL_VIEW = "Overlay all routes"
PANEL_VIEW = "Side-by-side panels"

st.set_page_config(
    page_title="Smart Campus Route Planner",
    page_icon="🧭",
    layout="wide",
)


@st.cache_resource
def get_graph():
    return load_graph()


@st.cache_data
def get_scenarios() -> List[dict]:
    return list(load_scenarios()["scenarios"])


def node_label(graph, node_id: str) -> str:
    node = graph.get_node(node_id)
    return (
        f"{node.name} · {node.building} · T{node.floor} · {node.node_type} ({node_id})"
    )


def short_name(graph, node_id: str) -> str:
    node = graph.get_node(node_id)
    return node.name if len(node.name) <= 30 else node.name[:29] + "…"


def edge_catalog(graph) -> Dict[str, Tuple[str, str]]:
    seen: Dict[Tuple[str, str], str] = {}
    for node_id in graph.node_ids:
        for edge in graph.neighbors(node_id):
            pair = tuple(sorted((edge.source, edge.target)))
            seen.setdefault(pair, f"{pair[0]} ↔ {pair[1]}")
    return {label: pair for pair, label in seen.items()}


def usable_edges(graph, current: str, following: str, blocked, accessible: bool) -> list:
    edges = []
    for edge in graph.neighbors(current):
        other = edge.target if edge.source == current else edge.source
        if other != following:
            continue
        if tuple(sorted((edge.source, edge.target))) in blocked:
            continue
        if accessible and not edge.accessible:
            continue
        edges.append(edge)
    return edges


def scenario_constraints(scenario: dict) -> Tuple[List[Tuple[str, str]], bool]:
    constraints = scenario.get("constraints") or {}
    blocked = [tuple(sorted(edge)) for edge in constraints.get("blocked_edges", [])]
    return blocked, bool(constraints.get("accessible_only", False))


def run_algorithm(
    graph,
    name: str,
    start: str,
    goal: str,
    blocked: List[Tuple[str, str]],
    accessible: bool,
) -> dict:
    stats: Dict[str, object] = {}
    started = time.perf_counter()
    try:
        if name == "BFS":
            path, expanded = student_core.bfs(
                graph,
                start,
                goal,
                blocked_edges=blocked,
                accessible_only=accessible,
                stats=stats,
            )
        elif name == "Greedy":
            path, expanded = student_core.greedy(
                graph,
                start,
                goal,
                student_core.euclidean_heuristic,
                blocked_edges=blocked,
                accessible_only=accessible,
                stats=stats,
            )
        else:
            path, expanded = student_core.astar(
                graph,
                start,
                goal,
                student_core.euclidean_heuristic,
                edge_weight="distance",
                blocked_edges=blocked,
                accessible_only=accessible,
                stats=stats,
            )
        error = None
    except Exception as exc:
        path, expanded, error = None, 0, f"{type(exc).__name__}: {exc}"
    elapsed = (time.perf_counter() - started) * 1000.0
    distance = (
        student_core.path_cost(graph, path, "distance", blocked, accessible)
        if path
        else None
    )
    seconds = (
        student_core.path_cost(graph, path, "travel_time", blocked, accessible)
        if path
        else None
    )
    return {
        "algorithm": name,
        "path": path,
        "expanded": int(stats.get("expanded", expanded)),
        "frontier_peak": int(stats.get("frontier_peak", 0)),
        "runtime_ms": elapsed,
        "distance": distance,
        "seconds": seconds,
        "error": error,
        "violations": path_violations(graph, path, blocked, accessible),
    }


def path_violations(
    graph,
    path: Optional[List[str]],
    blocked: List[Tuple[str, str]],
    accessible: bool,
) -> List[str]:
    if not path:
        return []
    blocked_set = {tuple(sorted(pair)) for pair in blocked}
    return [
        f"{current} → {nxt} has no usable corridor"
        for current, nxt in zip(path, path[1:])
        if not usable_edges(graph, current, nxt, blocked_set, accessible)
    ]


def leg_details(graph, path: Optional[List[str]], blocked, accessible) -> pd.DataFrame:
    if not path or len(path) < 2:
        return pd.DataFrame()
    blocked_set = {tuple(sorted(pair)) for pair in blocked}
    rows = []
    for index, (current, nxt) in enumerate(zip(path, path[1:]), start=1):
        candidates = usable_edges(graph, current, nxt, blocked_set, accessible)
        if not candidates:
            continue
        edge = min(candidates, key=lambda e: e.distance)
        rows.append(
            {
                "Step": index,
                "From": short_name(graph, current),
                "To": short_name(graph, nxt),
                "Distance (m)": round(edge.distance, 1),
                "Time (s)": round(edge.travel_time, 1),
                "Crowd": edge.crowd_level,
                "Stairs": edge.stairs,
                "Accessible": edge.accessible,
                "Indoor": edge.indoor,
            }
        )
    return pd.DataFrame(rows)


def route_steps_markdown(graph, path: Optional[List[str]]) -> str:
    if not path:
        return "No route."
    if len(path) == 1:
        return "Start and goal are the same landmark."
    return "  \n".join(
        f"**{index}.** {short_name(graph, current)} → {short_name(graph, nxt)}"
        for index, (current, nxt) in enumerate(zip(path, path[1:]), start=1)
    )


def results_frame(results: List[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "Algorithm": result["algorithm"],
                "Route": "found" if result["path"] else "no path",
                "Hops": len(result["path"]) - 1 if result["path"] else 0,
                "Distance (m)": (
                    round(result["distance"], 1)
                    if result["distance"] is not None
                    else None
                ),
                "Travel time (s)": (
                    round(result["seconds"], 1) if result["seconds"] is not None else None
                ),
                "Expanded": result["expanded"],
                "Frontier peak": result["frontier_peak"],
                "Runtime (ms)": round(result["runtime_ms"], 3),
                "Constraint check": (
                    "violation" if result["violations"] else ("ok" if result["path"] else "—")
                ),
            }
        )
    return pd.DataFrame(rows)


def draw_routes(graph, results, start, goal, blocked, accessible, title):
    routes = {
        result["algorithm"]: (result["path"], COLORS[result["algorithm"]])
        for result in results
    }
    return visuals.render_map(
        graph,
        routes=routes,
        blocked_edges=blocked,
        accessible_only=accessible,
        start=start,
        goal=goal,
        title=title,
    )


def scenario_state(graph, scenario: dict) -> dict:
    constraints = scenario.get("constraints") or {}
    blocked_keys = {
        tuple(sorted(edge)) for edge in constraints.get("blocked_edges", [])
    }
    labels = {label for label, pair in edge_catalog(graph).items() if pair in blocked_keys}
    return {
        "start": scenario["start"],
        "end": scenario["end"],
        "accessible": bool(constraints.get("accessible_only", False)),
        "blocks": labels,
    }


graph = get_graph()
scenarios = get_scenarios()
node_ids = sorted(graph.node_ids)
pairs = edge_catalog(graph)
edge_labels = sorted(pairs)

st.session_state.setdefault("start_node", "N_GATE")
st.session_state.setdefault("end_node", "N_D7_101")
st.session_state.setdefault("accessible_only", False)
st.session_state.setdefault("blocked_labels", set())
st.session_state.setdefault("block_version", 0)

pending = st.session_state.pop("pending_scenario", None)
if pending:
    st.session_state.start_node = pending["start"]
    st.session_state.end_node = pending["end"]
    st.session_state.accessible_only = pending["accessible"]
    st.session_state.blocked_labels = pending["blocks"]
    st.session_state.block_version += 1

st.title("Smart Campus Route Planner")
st.caption(
    "Hanoi University campus graph · 54 landmarks · BFS vs Greedy Best-First vs A* · "
    "accessibility-aware routing with live maintenance blocks"
)

with st.sidebar:
    st.header("Landmarks")
    start = st.selectbox(
        "Start landmark",
        node_ids,
        key="start_node",
        format_func=lambda nid: node_label(graph, nid),
    )
    goal = st.selectbox(
        "Destination landmark",
        node_ids,
        key="end_node",
        format_func=lambda nid: node_label(graph, nid),
    )

    st.header("Constraints")
    accessible = st.checkbox(
        "Accessible-only routing (step-free, lifts and flat corridors)",
        key="accessible_only",
    )

    blocked_selection = st.multiselect(
        "Blocked corridors (maintenance)",
        options=edge_labels,
        default=[label for label in edge_labels if label in st.session_state.blocked_labels],
        key=f"blocks_{st.session_state.block_version}",
        help="Every corridor selected here is removed from the graph for all algorithms.",
    )
    st.session_state.blocked_labels = set(blocked_selection)

    if st.button("Clear all maintenance blocks", use_container_width=True):
        st.session_state.blocked_labels = set()
        st.session_state.block_version += 1
        st.rerun()

    summary = graph.summary()
    st.divider()
    st.metric("Active blocks", len(st.session_state.blocked_labels))
    st.caption(
        f"Graph: {summary['num_nodes']} landmarks, {summary['num_edges']} corridors, "
        f"{len(summary['buildings'])} buildings/zones"
    )

blocked_pairs = [pairs[label] for label in sorted(st.session_state.blocked_labels)]
accessible_mode = st.session_state.accessible_only

planner_tab, stress_tab, bench_tab = st.tabs(
    ["Route planner", "Stress scenarios", "Algorithm benchmark"]
)

with planner_tab:
    if start == goal:
        st.info(
            "Start and destination are the same landmark. Pick a different destination "
            "to compute routes."
        )
    else:
        results = [
            run_algorithm(graph, name, start, goal, blocked_pairs, accessible_mode)
            for name in ALGORITHMS
        ]

        failed = [result["algorithm"] for result in results if not result["path"]]
        crashed = [result for result in results if result["error"]]
        violations = [result for result in results if result["violations"]]

        if crashed:
            for result in crashed:
                st.error(f"{result['algorithm']} failed to run: {result['error']}")
        if violations:
            st.error(
                "A returned route breaks an active constraint: "
                + ", ".join(result["algorithm"] for result in violations)
            )

        if len(failed) == len(results):
            st.error(
                "**No path found.** Every corridor usable under the current constraints is "
                "exhausted, so no route connects these landmarks. The planner reports this "
                "instead of returning an invalid route."
            )
            reasons = []
            if blocked_pairs:
                reasons.append(
                    f"{len(blocked_pairs)} corridor(s) blocked for maintenance"
                )
            if accessible_mode:
                reasons.append("accessible-only routing is enabled")
            if reasons:
                st.markdown(
                    "- Active restrictions: "
                    + "; ".join(reasons)
                    + ".\n- Try clearing a maintenance block on the main east–west spine, "
                    "or switch accessibility mode off if stairs are acceptable."
                )
        elif failed:
            st.warning(
                "No route for: "
                + ", ".join(failed)
                + ". The remaining algorithms returned a valid route."
            )

        map_view = st.radio(
            "Map view",
            [ALL_VIEW, PANEL_VIEW],
            horizontal=True,
            label_visibility="collapsed",
        )

        if map_view == ALL_VIEW:
            figure = draw_routes(
                graph,
                results,
                start,
                goal,
                blocked_pairs,
                accessible_mode,
                f"{short_name(graph, start)} → {short_name(graph, goal)}",
            )
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)
        else:
            for column, result in zip(st.columns(3), results):
                with column:
                    figure = visuals.render_map(
                        graph,
                        routes={
                            result["algorithm"]: (
                                result["path"],
                                COLORS[result["algorithm"]],
                            )
                        },
                        blocked_edges=blocked_pairs,
                        accessible_only=accessible_mode,
                        start=start,
                        goal=goal,
                        title=result["algorithm"],
                        figsize=(4.1, 3.3),
                    )
                    st.pyplot(figure, use_container_width=True)
                    plt.close(figure)

        st.subheader("Comparison")
        st.dataframe(
            results_frame(results).set_index("Algorithm"),
            use_container_width=True,
        )
        st.caption(
            "BFS ignores edge cost and minimises hops, so its distance can exceed the "
            "weighted optimum. Greedy best-first expands the fewest nodes but is not "
            "cost-optimal. A* is optimal on distance with an admissible, consistent "
            "Euclidean heuristic."
        )

        best = min(
            (result for result in results if result["path"]),
            key=lambda result: result["distance"] if result["distance"] is not None else float("inf"),
            default=None,
        )
        if best:
            st.success(
                f"Cheapest route: **{best['algorithm']}** · "
                f"{best['distance']:.1f} m · {best['seconds'] / 60:.1f} min walk · "
                f"{len(best['path']) - 1} hops"
            )

        st.subheader("Turn-by-turn directions")
        for name in ALGORITHMS:
            result = next(item for item in results if item["algorithm"] == name)
            hops = f"{len(result['path']) - 1} hops" if result["path"] else "no path"
            with st.expander(f"{name} — {hops}"):
                if result["error"]:
                    st.error(result["error"])
                elif not result["path"]:
                    st.warning("No route under the current constraints.")
                else:
                    st.markdown(route_steps_markdown(graph, result["path"]))
                    st.dataframe(
                        leg_details(graph, result["path"], blocked_pairs, accessible_mode),
                        use_container_width=True,
                        hide_index=True,
                    )

with stress_tab:
    st.subheader("Path recalculation under blocked corridors")
    st.caption(
        "Load a graded scenario into the planner, or run the whole suite to confirm the "
        "planner recalculates around a closed primary corridor and reports impossible "
        "requests without crashing."
    )

    for scenario in scenarios:
        constraints = scenario.get("constraints") or {}
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(
                    f"**{scenario['id']}** · class `{scenario['class']}` · "
                    f"{short_name(graph, scenario['start'])} → "
                    f"{short_name(graph, scenario['end'])}"
                )
                st.caption(scenario["description"])
                detail = []
                if constraints.get("blocked_edges"):
                    detail.append(
                        "blocked: "
                        + ", ".join(
                            f"{a} ↔ {b}" for a, b in constraints["blocked_edges"]
                        )
                    )
                if constraints.get("accessible_only"):
                    detail.append("accessible-only")
                if detail:
                    st.markdown("`" + " · ".join(detail) + "`")
            with right:
                if st.button("Load scenario", key=f"load_{scenario['id']}"):
                    st.session_state.pending_scenario = scenario_state(graph, scenario)
                    st.rerun()

    st.divider()
    st.subheader("Isolation stress test")
    st.caption(
        "Severs every corridor incident to the destination so the goal becomes "
        "unreachable. The planner must report no path and stay responsive."
    )
    isolate_col, start_col, _ = st.columns([1.4, 1.4, 2])
    with isolate_col:
        if st.button("Seal destination", use_container_width=True):
            sealed = {
                label
                for label, pair in pairs.items()
                if st.session_state.end_node in pair
            }
            st.session_state.blocked_labels = sealed
            st.session_state.block_version += 1
            st.rerun()
    with start_col:
        if st.button("Seal start landmark", use_container_width=True):
            sealed = {
                label
                for label, pair in pairs.items()
                if st.session_state.start_node in pair
            }
            st.session_state.blocked_labels = sealed
            st.session_state.block_version += 1
            st.rerun()

    if st.button("Run full stress suite", type="primary"):
        suite_rows = []
        for scenario in scenarios:
            blocked, accessible_flag = scenario_constraints(scenario)

            baseline = run_algorithm(
                graph, "A*", scenario["start"], scenario["end"], [], accessible_flag
            )
            constrained = run_algorithm(
                graph, "A*", scenario["start"], scenario["end"], blocked, accessible_flag
            )
            expects_failure = scenario["class"] == "failure"

            if constrained["error"]:
                verdict = "runtime error"
            elif expects_failure and not constrained["path"]:
                verdict = "no path reported safely"
            elif expects_failure and constrained["path"]:
                verdict = "unexpected route"
            elif constrained["path"] and not constrained["violations"]:
                verdict = "route recalculated"
            elif constrained["path"] and constrained["violations"]:
                verdict = "constraint violated"
            else:
                verdict = "no path"

            baseline_distance = baseline["distance"]
            new_distance = constrained["distance"]
            delta = (
                round(new_distance - baseline_distance, 1)
                if new_distance is not None and baseline_distance is not None
                else None
            )
            detour = (
                round(new_distance / baseline_distance, 2)
                if new_distance and baseline_distance
                else None
            )
            suite_rows.append(
                {
                    "Scenario": scenario["id"],
                    "Class": scenario["class"],
                    "Route found": bool(constrained["path"]),
                    "Baseline distance (m)": (
                        round(baseline_distance, 1) if baseline_distance is not None else None
                    ),
                    "Constrained distance (m)": (
                        round(new_distance, 1) if new_distance is not None else None
                    ),
                    "Δ distance (m)": delta,
                    "Detour ratio": detour,
                    "Expanded": constrained["expanded"],
                    "Verdict": verdict,
                }
            )

        suite = pd.DataFrame(suite_rows)
        st.dataframe(suite, use_container_width=True, hide_index=True)

        recalculated = suite["Verdict"].eq("route recalculated").sum()
        safe_failures = suite["Verdict"].eq("no path reported safely").sum()
        broken = suite["Verdict"].isin(
            ["runtime error", "constraint violated", "unexpected route"]
        ).sum()

        message = (
            f"{recalculated} scenario(s) recalculated a valid detour, "
            f"{safe_failures} impossibility case(s) reported cleanly, "
            f"{broken} failure(s)."
        )
        if int(broken) == 0:
            st.success(message)
        else:
            st.error(message)

        blocked_example = suite[suite["Scenario"] == "S2_blocked_edge"]
        if not blocked_example.empty:
            row = blocked_example.iloc[0]
            if pd.notna(row["Δ distance (m)"]):
                st.info(
                    "S2 with the south stadium spine closed: A* returns a detour of "
                    f"{row['Δ distance (m)']:+.1f} m "
                    f"({row['Detour ratio']:.2f}× the unconstrained distance) instead of "
                    "failing, proving the search re-plans after a primary corridor closes."
                )

        st.download_button(
            "Download stress results (CSV)",
            suite.to_csv(index=False).encode("utf-8"),
            file_name="stress_scenarios_results.csv",
            mime="text/csv",
        )

with bench_tab:
    st.subheader("Algorithm benchmark across all graded scenarios")
    st.caption(
        "BFS, greedy best-first and A* are run on every scenario with its own constraints "
        "so search effort and solution quality can be compared directly."
    )

    if st.button("Run benchmark", type="primary"):
        rows = []
        for scenario in scenarios:
            blocked, accessible_flag = scenario_constraints(scenario)
            for name in ALGORITHMS:
                result = run_algorithm(
                    graph, name, scenario["start"], scenario["end"], blocked, accessible_flag
                )
                rows.append(
                    {
                        "Scenario": scenario["id"],
                        "Algorithm": name,
                        "Path cost": (
                            round(result["distance"], 1)
                            if result["distance"] is not None
                            else None
                        ),
                        "Expanded": result["expanded"],
                        "Frontier peak": result["frontier_peak"],
                        "Runtime (ms)": round(result["runtime_ms"], 3),
                    }
                )
        frame = pd.DataFrame(rows)
        st.session_state.benchmark_frame = frame

        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download benchmark (CSV)",
            frame.to_csv(index=False).encode("utf-8"),
            file_name="algorithm_benchmark.csv",
            mime="text/csv",
        )

    frame = st.session_state.get("benchmark_frame")
    if frame is not None and not frame.empty:
        metric = st.selectbox(
            "Chart metric",
            ["Expanded", "Frontier peak", "Runtime (ms)", "Path cost"],
        )
        chart_data = frame.dropna(subset=[metric])
        if chart_data.empty:
            st.info("No path exists for this metric under the graded scenarios.")
        else:
            figure = visuals.render_benchmark_chart(chart_data, metric)
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)

        st.subheader("Findings")
        finding_rows = []
        for scenario_id in frame["Scenario"].unique():
            subset = frame[frame["Scenario"] == scenario_id].set_index("Algorithm")
            cheapest = subset["Path cost"].idxmin() if subset["Path cost"].notna().any() else None
            fewest = subset["Expanded"].idxmin() if subset["Expanded"].notna().any() else None
            finding_rows.append(
                {
                    "Scenario": scenario_id,
                    "Cheapest route": cheapest,
                    "Fewest nodes expanded": fewest,
                    "Expansion ratio BFS/A*": (
                        round(float(subset.loc["BFS", "Expanded"]) / float(subset.loc["A*", "Expanded"]), 2)
                        if "BFS" in subset.index
                        and "A*" in subset.index
                        and float(subset.loc["A*", "Expanded"]) > 0
                        else None
                    ),
                }
            )
        st.dataframe(pd.DataFrame(finding_rows), use_container_width=True, hide_index=True)
