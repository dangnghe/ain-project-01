from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))

from graph import load_graph, load_scenarios  # noqa: E402
import student_core as core  # noqa: E402
import visuals  # noqa: E402

PAGE_W, PAGE_H = 8.5, 11.0
MARGIN = 0.72
COL_GAP = 0.26
COL_W = (PAGE_W - 2 * MARGIN - COL_GAP) / 2.0
BODY = 9.0
FONT = "STIXGeneral"

plt.rcParams.update(
    {
        "font.family": FONT,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    }
)

MEASURE_FIG = plt.figure(figsize=(2.0, 1.0), dpi=72)
MEASURE = MEASURE_FIG.canvas.get_renderer()


def prop_for(size: float, weight: str = "normal", style: str = "normal") -> FontProperties:
    return FontProperties(family=FONT, size=size, weight=weight, style=style)


def text_width(text: str, prop: FontProperties) -> float:
    if not text:
        return 0.0
    artist = Text(0.0, 0.0, text, fontproperties=prop, figure=MEASURE_FIG)
    return artist.get_window_extent(renderer=MEASURE).width / 72.0


def tokens(text: str) -> List[str]:
    out: List[str] = []
    current = ""
    in_math = False
    for char in text:
        if char == "$":
            in_math = not in_math
            current += char
            continue
        if char == " " and not in_math:
            if current:
                out.append(current)
                current = ""
            continue
        current += char
    if current:
        out.append(current)
    return out


class Paper:
    def __init__(self, path: Path):
        self.pdf = PdfPages(str(path))
        self.fig = None
        self.page_no = 0
        self.col = 0
        self.y = 0.0
        self.first_heading_done = False
        self._start_page()

    def _start_page(self) -> None:
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.fig.patch.set_facecolor("white")
        self.page_no += 1
        self.col = 0
        self.y = PAGE_H - MARGIN
        self.body_top = PAGE_H - MARGIN
        self._t(PAGE_W / 2.0, 0.42, str(self.page_no), size=8.0, ha="center")

    def _close_page(self) -> None:
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def new_column(self) -> None:
        if self.col == 0:
            self.col = 1
            self.y = self.body_top
            return
        self._close_page()
        self._start_page()

    def x0(self) -> float:
        return MARGIN + self.col * (COL_W + COL_GAP)

    def _t(
        self,
        x: float,
        y: float,
        text: str,
        size: float = BODY,
        weight: str = "normal",
        style: str = "normal",
        ha: str = "left",
    ) -> None:
        self.fig.text(
            x / PAGE_W,
            y / PAGE_H,
            text,
            fontfamily=FONT,
            fontsize=size,
            fontweight=weight,
            fontstyle=style,
            ha=ha,
            va="baseline",
        )

    def wrap(self, text: str, width: float, prop: FontProperties) -> List[List[str]]:
        lines: List[List[str]] = []
        current: List[str] = []
        for word in tokens(text):
            trial = " ".join(current + [word])
            if current and text_width(trial, prop) > width:
                lines.append(current)
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(current)
        return lines

    def _draw_lines(
        self,
        lines: List[List[str]],
        x: float,
        width: float,
        prop: FontProperties,
        size: float,
        leading: float,
        justify: bool,
        align: str = "left",
        style: str = "normal",
    ) -> None:
        for index, line in enumerate(lines):
            if not line:
                self.y -= leading
                continue
            last = index == len(lines) - 1
            rendered = " ".join(line)
            natural = sum(text_width(token, prop) for token in line) + text_width(" ", prop) * (len(line) - 1)
            line_justify = justify and not any("$" in token for token in line)
            if natural < 0.72 * width:
                line_justify = False
            if align == "center":
                self._t(x + width / 2.0, self.y, rendered, size=size, ha="center", style=style)
            elif line_justify and not last and len(line) > 1:
                widths = [text_width(word, prop) for word in line]
                gap = (width - sum(widths)) / (len(line) - 1)
                if gap > 0:
                    cursor = x
                    for word, word_width in zip(line, widths):
                        self._t(cursor, self.y, word, size=size, style=style)
                        cursor += word_width + gap
                else:
                    self._t(x, self.y, rendered, size=size, style=style)
            else:
                self._t(x, self.y, rendered, size=size, style=style)
            self.y -= leading

    def space(self, amount: float = 0.08) -> None:
        self.y -= amount

    def title_block(self, title: str, authors: str, affiliation: str, email: str) -> None:
        self.y -= 0.10
        width = PAGE_W - 2 * MARGIN
        prop = prop_for(19.0, "bold")
        lines = self.wrap(title, width * 0.92, prop)
        for line in lines:
            self._t(PAGE_W / 2.0, self.y, " ".join(line), size=19.0, weight="bold", ha="center")
            self.y -= 0.31
        self.y -= 0.06
        for line in authors.split("\n"):
            self._t(PAGE_W / 2.0, self.y, line, size=11.0, ha="center")
            self.y -= 0.19
        self.y -= 0.02
        for line in affiliation.split("\n"):
            self._t(PAGE_W / 2.0, self.y, line, size=9.0, style="italic", ha="center")
            self.y -= 0.16
        self._t(PAGE_W / 2.0, self.y, email, size=9.0, ha="center")
        self.y -= 0.30
        self.body_top = self.y

    def wide_paragraph(self, text: str, size: float = 9.0, label: str = "", style: str = "normal") -> None:
        width = PAGE_W - 2 * MARGIN
        prop = prop_for(size, style=style)
        lines = self.wrap(text, width, prop)
        leading = size * 1.18 / 72.0
        if label:
            self._t(MARGIN, self.y, label, size=size, weight="bold")
            offset = text_width(label + " ", prop_for(size, weight="bold"))
            self._draw_lines([lines[0]], MARGIN + offset, width - offset, prop, size, leading, justify=False, style=style)
            for line in lines[1:]:
                self._t(MARGIN, self.y, " ".join(line), size=size, style=style)
                self.y -= leading
            return
        self._draw_lines(lines, MARGIN, width, prop, size, leading, justify=True, style=style)

    def heading(self, number: str, title: str) -> None:
        if self.page_no == 1 and self.col == 0 and not self.first_heading_done:
            self.body_top = self.y
            self.first_heading_done = True
        if self.y - 2.60 < MARGIN:
            self.new_column()
        self.y -= 0.14
        label = f"{number}. {title.upper()}" if number else title.upper()
        self._t(self.x0() + COL_W / 2.0, self.y, label, size=8.5, weight="bold", ha="center")
        self.y -= 0.22

    def subheading(self, title: str) -> None:
        if self.y - 0.85 < MARGIN:
            self.new_column()
        self.y -= 0.09
        self._t(self.x0(), self.y, title, size=8.6, style="italic")
        self.y -= 0.17

    def paragraph(self, text: str, size: float = BODY, indent: float = 0.19) -> None:
        leading = size * 1.18 / 72.0
        prop = prop_for(size)
        width = COL_W - indent
        lines = self.wrap(text, width, prop)
        if self.y - len(lines) * leading < MARGIN:
            self.new_column()
        self._draw_lines(lines, self.x0() + indent, width, prop, size, leading, justify=True)

    def bullet(self, text: str, size: float = BODY) -> None:
        leading = size * 1.18 / 72.0
        prop = prop_for(size)
        width = COL_W - 0.26
        lines = self.wrap(text, width, prop)
        if self.y - len(lines) * leading < MARGIN:
            self.new_column()
        self._t(self.x0() + 0.08, self.y, "•", size=size)
        self._draw_lines(lines, self.x0() + 0.26, width, prop, size, leading, justify=False)
        self.y -= 0.02

    def equation(self, latex: str, number: str = "") -> None:
        if self.y - 0.34 < MARGIN:
            self.new_column()
        self.y -= 0.17
        self._t(self.x0() + COL_W / 2.0 - 0.14, self.y, latex, size=9.0, ha="center")
        if number:
            self._t(self.x0() + COL_W, self.y, f"({number})", size=9.0, ha="right")
        self.y -= 0.17

    def figure(self, path: Path, caption: str) -> None:
        image = plt.imread(str(path))
        height = COL_W * image.shape[0] / image.shape[1]
        caption_prop = prop_for(7.6, style="italic")
        caption_lines = self.wrap(caption, COL_W, caption_prop)
        caption_height = len(caption_lines) * 0.115
        total = height + caption_height + 0.16
        if self.y - total < MARGIN:
            self.new_column()
        self.y -= 0.09
        axes = self.fig.add_axes([self.x0() / PAGE_W, (self.y - height) / PAGE_H, COL_W / PAGE_W, height / PAGE_H])
        axes.imshow(image)
        axes.axis("off")
        self.y -= height + 0.09
        self._draw_lines(caption_lines, self.x0(), COL_W, caption_prop, 7.6, 0.115, justify=False, align="center", style="italic")
        self.y -= 0.02

    def table(self, tag: str, title: str, headers: List[str], rows: List[List[str]], weights: List[float], size: float = 7.4) -> None:
        total_weight = sum(weights)
        widths = [COL_W * weight / total_weight for weight in weights]
        leading = size * 1.32 / 72.0
        height = 0.16 + 0.14 + (len(rows) + 1) * leading + 0.16
        if self.y - height < MARGIN:
            self.new_column()
        self.y -= 0.12
        self._t(self.x0() + COL_W / 2.0, self.y, tag, size=7.4, ha="center")
        self.y -= 0.14
        self._t(self.x0() + COL_W / 2.0, self.y, title, size=7.4, ha="center")
        self.y -= 0.10
        top = self.y

        def rule(y: float, width_pt: float = 0.6) -> None:
            self.fig.add_artist(
                Line2D(
                    [self.x0() / PAGE_W, (self.x0() + COL_W) / PAGE_W],
                    [y / PAGE_H, y / PAGE_H],
                    color="black",
                    linewidth=width_pt,
                    transform=self.fig.transFigure,
                )
            )

        rule(top, 1.0)
        self.y -= 0.11
        cursor = self.x0()
        for header, width in zip(headers, widths):
            self._t(cursor + width / 2.0, self.y, header, size=size, ha="center")
            cursor += width
        self.y -= 0.05
        rule(self.y + 0.04, 0.6)
        self.y -= 0.10
        for row in rows:
            cursor = self.x0()
            for index, (cell, width) in enumerate(zip(row, widths)):
                if index == 0:
                    self._t(cursor + 0.03, self.y, cell, size=size)
                else:
                    self._t(cursor + width / 2.0, self.y, cell, size=size, ha="center")
                cursor += width
            self.y -= leading
        rule(self.y + 0.045, 1.0)
        self.y -= 0.06

    def finish(self) -> None:
        self._close_page()
        self.pdf.close()


def measure(graph, name: str, start: str, goal: str, blocked, accessible: bool) -> dict:
    timings: List[float] = []
    stats: Dict[str, object] = {}
    path = None
    for _ in range(60):
        stats.clear()
        began = time.perf_counter()
        if name == "BFS":
            path, expanded = core.bfs(graph, start, goal, blocked_edges=blocked, accessible_only=accessible, stats=stats)
        elif name == "Greedy":
            path, expanded = core.greedy(graph, start, goal, core.euclidean_heuristic, blocked_edges=blocked, accessible_only=accessible, stats=stats)
        else:
            path, expanded = core.astar(graph, start, goal, core.euclidean_heuristic, edge_weight="distance", blocked_edges=blocked, accessible_only=accessible, stats=stats)
        timings.append((time.perf_counter() - began) * 1000.0)
    cost = core.path_cost(graph, path, "distance", blocked, accessible) if path else None
    return {
        "path": path,
        "cost": cost,
        "expanded": int(stats.get("expanded", expanded)),
        "frontier": int(stats.get("frontier_peak", 0)),
        "generated": int(stats.get("generated", 0)),
        "ms": statistics.median(timings),
    }


def heuristic_astar(graph, start: str, goal: str, heuristic_name: str) -> dict:
    heuristic = core.HEURISTICS[heuristic_name]
    stats: Dict[str, object] = {}
    timings: List[float] = []
    path = None
    for _ in range(60):
        stats.clear()
        began = time.perf_counter()
        path, _ = core.astar(graph, start, goal, heuristic, edge_weight="distance", stats=stats)
        timings.append((time.perf_counter() - began) * 1000.0)
    return {
        "cost": core.path_cost(graph, path, "distance"),
        "expanded": int(stats.get("expanded", 0)),
        "ms": statistics.median(timings),
    }


graph = load_graph()
scenarios = load_scenarios()["scenarios"]
algorithms = ["BFS", "Greedy", "A*"]
ABBR = {"BFS": "BFS", "Greedy": "GBFS", "A*": "A*"}

results: Dict[str, Dict[str, dict]] = {}
for scenario in scenarios:
    constraints = scenario.get("constraints") or {}
    blocked = [tuple(edge) for edge in constraints.get("blocked_edges", [])]
    accessible = bool(constraints.get("accessible_only", False))
    results[scenario["id"]] = {
        name: measure(graph, name, scenario["start"], scenario["end"], blocked, accessible)
        for name in algorithms
    }

heuristics_s4 = {name: heuristic_astar(graph, "N_A1_PARKING", "N_D6_101", name) for name in ("euclidean", "manhattan")}

baseline_s1 = results["S1_normal_route"]["A*"]["cost"]
blocked_s2 = results["S2_blocked_edge"]["A*"]["cost"]
detour = blocked_s2 - baseline_s1
detour_ratio = blocked_s2 / baseline_s1
s4_astar = results["S4_heuristic_comparison"]
s4_bfs = s4_astar["BFS"]["cost"]
s4_astar_cost = s4_astar["A*"]["cost"]
s4_gain = s4_bfs - s4_astar_cost
s4_gain_pct = 100.0 * s4_gain / s4_bfs
expansion_ratio_s4 = s4_astar["BFS"]["expanded"] / s4_astar["A*"]["expanded"]
mean_bfs = statistics.mean(results[sid]["BFS"]["expanded"] for sid in results if results[sid]["BFS"]["path"])
mean_astar = statistics.mean(results[sid]["A*"]["expanded"] for sid in results if results[sid]["A*"]["path"])
mean_gbfs = statistics.mean(results[sid]["Greedy"]["expanded"] for sid in results if results[sid]["Greedy"]["path"])

tmp = Path(tempfile.mkdtemp(prefix="report_assets_"))

frame_rows = []
for scenario in scenarios:
    for name in algorithms:
        outcome = results[scenario["id"]][name]
        frame_rows.append(
            {
                "Scenario": scenario["id"].split("_")[0],
                "Algorithm": name,
                "Path cost": outcome["cost"] if outcome["cost"] is not None else float("nan"),
                "Expanded": outcome["expanded"],
                "Frontier peak": outcome["frontier"],
                "Runtime (ms)": outcome["ms"],
            }
        )
frame = pd.DataFrame(frame_rows)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

map_figure = visuals.render_map(
    graph,
    routes={
        "Unconstrained route (S1)": (results["S1_normal_route"]["A*"]["path"], "#2f9e44"),
        "Recalculated route (S2)": (results["S2_blocked_edge"]["A*"]["path"], "#e8590c"),
    },
    blocked_edges=[("W_CROSS_EAST", "W_SOUTH_STADIUM")],
    start="N_GATE",
    goal="N_D7_101",
    title="S2: closed south stadium spine and the recalculated detour",
    figsize=(7.6, 5.2),
)
map_path = tmp / "map_s1_s2.png"
map_figure.savefig(map_path, dpi=260, bbox_inches="tight", facecolor=visuals.BACKGROUND)
plt.close(map_figure)

chart_data = frame[frame["Path cost"].notna()]
expansion_chart = visuals.render_benchmark_chart(chart_data, "Expanded")
expansion_path = tmp / "expanded.png"
expansion_chart.savefig(expansion_path, dpi=260, bbox_inches="tight", facecolor=visuals.BACKGROUND)
plt.close(expansion_chart)

cost_chart = visuals.render_benchmark_chart(chart_data, "Path cost")
cost_path = tmp / "cost.png"
cost_chart.savefig(cost_path, dpi=260, bbox_inches="tight", facecolor=visuals.BACKGROUND)
plt.close(cost_chart)

euclid = heuristics_s4["euclidean"]
manhattan = heuristics_s4["manhattan"]
manhattan_penalty = 100.0 * (manhattan["cost"] - euclid["cost"]) / euclid["cost"]

plt.rcParams["font.family"] = FONT

paper = Paper(ROOT / "report.pdf")

paper.title_block(
    "Constraint-Aware Campus Wayfinding: An Interactive Implementation and Evaluation of BFS, Greedy Best-First, and A{*} Search".replace("{*}", "*"),
    "Author Name",
    "Department of Information Technology\nHanoi University, Hanoi, Vietnam",
    "author@example.com",
)

paper.wide_paragraph(
    "Campus wayfinding is a constrained shortest-path problem in which routes must satisfy "
    "accessibility and temporary-closure requirements while remaining cheap in distance and "
    "walking time. This paper reports the design, implementation and empirical evaluation of an "
    f"interactive route-planning product built on a 54-landmark, 75-corridor campus graph. Three "
    "classical search algorithms, breadth-first search (BFS), greedy best-first search (GBFS) and "
    "A*, are implemented from first principles and compared on nodes expanded, peak frontier "
    f"size, solution cost and runtime across five graded scenarios. A* with the Euclidean "
    f"heuristic expands {expansion_ratio_s4:.1f}\u00d7 fewer nodes than BFS and returns routes that are "
    f"{s4_gain:.1f} m ({s4_gain_pct:.1f} %) cheaper in the heuristic-comparison scenario, while GBFS "
    f"expands the smallest frontier but is not cost-optimal. The product exposes accessibility-only "
    "routing, live maintenance blocks and side-by-side route comparison in a Streamlit interface, "
    "and a stress suite demonstrates that the planner relocates a route around a closed primary "
    f"corridor with a {detour:+.1f} m detour ({detour_ratio:.2f}\u00d7 baseline) and reports "
    "unreachable requests explicitly instead of crashing or returning an invalid path.",
    size=9.0,
    label="Abstract\u2014",
    style="italic",
)
paper.space(0.05)
paper.wide_paragraph(
    "campus navigation, A* search, greedy best-first search, admissible heuristic, "
    "accessibility-aware routing, interactive path planning.",
    size=9.0,
    label="Index Terms\u2014",
    style="italic",
)
paper.space(0.10)

paper.heading("I", "Introduction")
paper.paragraph(
    "Modern university campuses are multi-building environments stitched together by walkways, "
    "staircases, lifts and plazas. Wayfinding inside them is not a simple geometric problem: a "
    "corridor can be closed for maintenance, a staircase is unusable for a visitor with reduced "
    "mobility, and a route that is shortest in metres may be slower than a slightly longer route in "
    "a less crowded arcade. A planner that ignores these constraints produces paths that are "
    "formally valid but practically useless, which is exactly the failure mode a navigation product "
    "must avoid."
)
paper.paragraph(
    "This work formulates campus wayfinding as a state-space search over an undirected weighted "
    "graph and delivers a working product rather than an isolated algorithm. The contributions are:"
)
paper.bullet(
    "A from-scratch implementation of BFS, greedy best-first search and A*, together with "
    "Euclidean and Manhattan heuristics, that accepts dynamic constraints (blocked corridor pairs "
    "and accessibility-only traversal) on every query."
)
paper.bullet(
    "An interactive Streamlit application in which a user selects start and destination landmarks, "
    "toggles step-free routing, blocks corridors to simulate maintenance, and compares all three "
    "algorithms on one map and in a metric table."
)
paper.bullet(
    "A stress suite that reloads the graded scenarios, quantifies the detour produced when a "
    "primary corridor is closed, and proves that impossible requests are detected and reported "
    "without exceptions or constraint violations."
)
paper.bullet(
    "A comparative evaluation on five scenarios reporting expansion counts, peak frontier sizes, "
    "route cost and runtime for every algorithm."
)

paper.heading("II", "Related Work")
paper.paragraph(
    "Uninformed and informed search form the classical core of artificial intelligence "
    "textbooks [1]. Breadth-first search explores the state space in layers and returns a path with "
    "the fewest edges, but it is blind to edge cost and expands a large fraction of the graph. "
    "Greedy best-first search orders the frontier by an estimate of the remaining distance [3]; "
    "it is fast and memory-friendly but offers no optimality guarantee. A*, introduced by Hart, "
    "Nilsson and Raphael [2], combines the accumulated cost with the estimate, and is optimal "
    "whenever the heuristic never overestimates the true remaining cost."
)
paper.paragraph(
    "Published campus-navigation systems usually couple such a search with a map front end and "
    "with accessibility annotations on edges. The product described here follows the same division "
    "of labour but keeps every algorithmic decision visible: the graph loader only exposes "
    "neighbours and attributes, and all search logic is written by hand in a single module, without "
    "a third-party graph library. That constraint makes the trade-offs measurable instead of "
    "hidden, which is the purpose of the evaluation in Section VI."
)

paper.heading("III", "Problem Formulation")
paper.paragraph(
    "The campus is a finite undirected graph $G = (V, E)$ with $|V| = 54$ landmarks and $|E| = 75$ "
    "corridors. Each vertex stores a name, a schematic coordinate, a floor, a building and a node "
    "type. Each edge stores a metric distance, a realistic travel time, a crowd level, and three "
    "booleans: stairs-only, indoor, and accessible. Because a stairs connector and a lift connector "
    "may join the same pair of floor nodes, parallel edges are preserved as distinct rows and the "
    "adjacency list keeps every incident edge."
)
paper.paragraph(
    "A query is a tuple $(s, t, B, a)$: a start landmark $s$, a destination $t$, a set $B$ of "
    "undirected corridor pairs closed for maintenance, and an accessibility flag $a$. The action "
    "set from a vertex $u$ contains every edge $(u, v)$ that is neither in $B$ nor, when $a$ is "
    "true, marked inaccessible. A path is a sequence of vertices joined by admissible actions, and "
    "its cost is the sum of edge weights:"
)
paper.equation(r"$g(\pi) = \sum_{(u,v)\in\pi} \; c(u,v)$", "1")
paper.paragraph(
    "where $c$ is either metric distance or travel time. The search minimises $g$ subject to the "
    "two dynamic constraints, which is why every algorithm in the product receives $B$ and $a$ as "
    "first-class parameters rather than filtering a pre-built adjacency list."
)
paper.paragraph(
    "Two geometric heuristics are provided. With $p(n)$ the schematic coordinates of a node and "
    "the goal fixed, the Euclidean and Manhattan estimates are"
)
paper.equation(r"$h_{e}(n) = \| p(n) - p(t) \|_{2}, \qquad h_{m}(n) = |\Delta x| + |\Delta y|$", "2")
paper.paragraph(
    "Admissibility ($h(n) \leq h^{*}(n)$) holds because every corridor length equals the "
    "Euclidean distance between its endpoints, so the straight-line distance can never exceed the "
    "length of any multi-hop detour. Consistency ($h(n) \leq c(n, n') + h(n')$) follows from the "
    "triangle inequality: $h(n) \leq |p(n) - p(n')| + h(n') = c(n, n') + h(n')$ for "
    "every edge. When the cost function is travel time instead of distance, both heuristics are "
    "scaled by the minimum observed time-per-metre ratio of the graph, which preserves both "
    "properties."
)

paper.paragraph(
    "The Manhattan estimate is admissible only when movement is restricted to axis-aligned steps. "
    "This campus graph contains diagonal corridors, so $h_m$ can exceed the true remaining cost; "
    "A* guided by it is therefore not guaranteed optimal, and Section VI-A measures that loss."
)

paper.heading("IV", "Algorithms")
paper.subheading("A. Breadth-First Search")
paper.paragraph(
    "BFS keeps a FIFO queue and a visited set, expanding vertices in non-decreasing hop count. It "
    "returns the path with the fewest corridors, which is the correct notion of optimality only "
    "when every edge has unit cost. Accessible routing and maintenance blocks are applied inside "
    "the neighbour expansion, so the search simply never sees a forbidden edge. Its memory use is "
    "bounded by the frontier width, the quantity the peak-frontier metric measures."
)
paper.subheading("B. Greedy Best-First Search")
paper.paragraph(
    "GBFS replaces the queue with a priority queue ordered by $h(n)$ alone and keeps a closed set "
    "to avoid re-expansion. It drives the search straight toward the goal and therefore expands "
    "very few vertices, but the first path it completes is not necessarily the cheapest: a locally "
    "attractive corridor can lead into an expensive approach to the destination. The product "
    "reports this honestly and labels the algorithm as non-optimal."
)
paper.subheading("C. A* Search")
paper.paragraph(
    "A* orders the frontier by $f(n) = g(n) + h(n)$ and maintains the best known cost per "
    "vertex. Because the Euclidean heuristic is consistent, a closed set may be used without "
    "re-opening vertices, and the first expansion of the goal yields an optimal route. Ties are "
    "broken by an insertion counter so the behaviour is deterministic across runs, which matters "
    "for a demo in which the audience compares metrics between reloads."
)
paper.paragraph(
    "All three implementations return a pair (path or None, number of expanded nodes) and "
    "optionally fill a statistics dictionary with expanded, peak-frontier and generated counts. "
    "The same contract makes the product's comparison view a thin loop over the three functions."
)

paper.heading("V", "System Architecture")
paper.paragraph(
    "The repository separates provided infrastructure from authored logic. The graph loader reads "
    "the two CSV files, validates that every edge references a known vertex and exposes typed "
    "neighbour queries. The search core owns every algorithmic decision. A rendering module draws "
    "the schematic map with matplotlib, and the product layer is a Streamlit application that "
    "recomposes those pieces into three workspaces."
)
paper.bullet(
    "Route planner. Landmark selectors, an accessibility switch and a multi-select of corridors "
    "form the constraint state. The three searches run on every interaction, and the results are "
    "shown as an overlaid map or as three side-by-side panels, followed by a metric table and "
    "turn-by-turn directions with per-leg distance, time, crowd level and stairs flags."
)
paper.bullet(
    "Stress scenarios. Each graded scenario can be loaded into the planner with one click; the "
    "panel also offers isolation controls that sever every corridor incident to the start or the "
    "destination, and a full suite that recomputes baseline and constrained routes for every "
    "scenario and classifies the outcome."
)
paper.bullet(
    "Algorithm benchmark. All three algorithms are run on all five scenarios and the resulting "
    "expansion, frontier and cost series are plotted and exportable as CSV."
)
paper.paragraph(
    "Interactive state is held in a small number of explicit session keys: start, destination, "
    "accessibility flag, the set of blocked corridor labels and a version counter. Loading a "
    "scenario stages the request in a pending record that is applied at the top of the script "
    "before any widget is instantiated, which keeps widget-backed state consistent and avoids the "
    "classic Streamlit error of mutating a key after its widget exists."
)
paper.paragraph(
    "Rendering is decoupled from search. The map routine accepts a dictionary from algorithm name "
    "to (path, colour), the blocked pairs and the accessibility flag, and draws blocked corridors "
    "as red dashed strokes, stairs-only corridors as dashed amber strokes and each route as a "
    "thick polyline. Because the drawing code never inspects the search results beyond the path, a "
    "missing route simply does not appear, and the failure is communicated by the table and the "
    "message banners instead of by a broken figure."
)
paper.paragraph(
    f"The computational profile is modest: the graph has 54 vertices, and the most expensive query "
    f"expands at most {max(r['expanded'] for sid in results for r in results[sid].values())} "
    f"vertices with a median runtime below "
    f"{max(r['ms'] for sid in results for r in results[sid].values()):.2f} ms. Each interaction "
    "therefore re-solves all three searches immediately, with no caching that could mask a stale "
    "constraint."
)

paper.heading("VI", "Experimental Evaluation")
paper.paragraph(
    "Every scenario is executed for all three algorithms with its own constraints. Table I reports "
    "the route cost, expanded nodes, peak frontier and median runtime over 60 repetitions. Two "
    "rows stand out immediately. In S4 the BFS route costs "
    f"{s4_bfs:.1f} m while A* finds {s4_astar_cost:.1f} m, a {s4_gain_pct:.1f} % improvement, "
    f"because BFS minimises hops rather than metres; GBFS matches the BFS cost while expanding "
    f"{s4_astar['Greedy']['expanded']} vertices instead of {s4_astar['BFS']['expanded']}. Across the "
    f"scenarios where a route exists, BFS expands {mean_bfs:.1f} vertices on average, GBFS "
    f"{mean_gbfs:.1f} and A* {mean_astar:.1f}."
)

table_rows = []
for scenario in scenarios:
    for name in algorithms:
        outcome = results[scenario["id"]][name]
        table_rows.append(
            [
                scenario["id"].split("_")[0],
                ABBR[name],
                f"{outcome['cost']:.1f}" if outcome["cost"] is not None else "none",
                str(outcome["expanded"]),
                str(outcome["frontier"]),
                f"{outcome['ms']:.2f}",
            ]
        )
paper.table(
    "TABLE I",
    "Route cost (m), expanded nodes, peak frontier and median runtime per scenario",
    ["Sc.", "Alg.", "Cost", "Exp.", "Fr.", "ms"],
    table_rows,
    [0.9, 0.9, 1.1, 0.9, 0.8, 1.0],
)

paper.paragraph(
    "The gap between BFS and A* is largest where a cheap long corridor competes with a short "
    "expensive one: in S4 the hop-minimal route walks "
    f"{s4_bfs:.1f} m, whereas A* accepts a route with {len(s4_astar['A*']['path']) - 1} "
    "corridors and a lower metric cost. This is the practical argument for an informed search in "
    "a walking product, since users experience metres and minutes, not hop counts."
)

paper.figure(
    map_path,
    "Fig. 1. Campus graph for S1 and S2. The red dashed corridor is the closed south stadium "
    "spine; the dashed amber corridors are stairs-only, the grey ones are step-free. Green is the "
    "unconstrained route and orange the route recalculated after the closure.",
)

paper.subheading("A. Heuristic Comparison")
paper.paragraph(
    "Scenario S4 compares the two heuristics with A* on the same query. Table II shows the "
    "practical cost of an inadmissible estimate: because diagonal corridors make the Manhattan "
    "estimate exceed the true remaining distance, A* settles for a route of "
    f"{manhattan['cost']:.1f} m instead of the {euclid['cost']:.1f} m reached with the Euclidean "
    f"heuristic, a {manhattan_penalty:.1f} % penalty. It expands only "
    f"{manhattan['expanded']} vertices against {euclid['expanded']} because an inflated estimate "
    "drives the search more aggressively toward the goal. The comparison isolates the trade-off "
    "that matters in a walking product: fewer expansions are worthless if the delivered route is "
    "longer than necessary."
)

paper.table(
    "TABLE II",
    "A* on scenario S4 with the Euclidean and Manhattan heuristics",
    ["Heuristic", "Cost (m)", "Expanded", "ms"],
    [
        ["Euclidean", f"{euclid['cost']:.1f}", str(euclid["expanded"]), f"{euclid['ms']:.2f}"],
        ["Manhattan", f"{manhattan['cost']:.1f}", str(manhattan["expanded"]), f"{manhattan['ms']:.2f}"],
    ],
    [2.0, 1.2, 1.1, 1.0],
)

paper.figure(
    cost_path,
    "Fig. 2. Path cost per scenario and algorithm. BFS is blind to cost and can return a hop-minimal "
    "route that is measurably longer; A* and GBFS coincide where the geometric order happens to "
    "agree with the cost order.",
)

paper.figure(
    expansion_path,
    "Fig. 3. Nodes expanded per scenario and algorithm. GBFS shows the smallest frontier in most "
    "cases, A* sits between the two, and BFS expands a large share of the graph in every "
    "scenario because it has no goal direction.",
)

paper.heading("VII", "Stress Testing and Failure Handling")
paper.paragraph(
    "The second half of the product specification concerns behaviour under adverse conditions, so "
    "the planner was tested against route loss and against impossible requests rather than only "
    "against happy paths."
)
paper.subheading("A. Recalculation After a Primary Corridor Closes")
paper.paragraph(
    "S2 closes the south stadium spine that the unconstrained route uses. All three algorithms "
    "return a valid detour instead of failing: the A* cost rises from "
    f"{baseline_s1:.1f} m to {blocked_s2:.1f} m, a {detour:+.1f} m change "
    f"({detour_ratio:.2f}\u00d7). The detour uses the northern stadium loop, which the map view "
    "makes visible directly. Because blocking is expressed as a set of corridor pairs and applied "
    "inside neighbour expansion, re-planning requires no rebuilding of the graph and no restart of "
    "the application; the new route appears on the next rerun of the interface."
)
paper.subheading("B. Detecting Impossible Requests")
paper.paragraph(
    "S5 asks for the technical attic of building C with accessibility enabled. The attic is "
    "reachable only through a narrow spiral staircase flagged stairs-only and inaccessible, so no "
    "step-free route exists. A*, GBFS and BFS all exhaust their frontiers and return the "
    "empty result, which the interface renders as an explicit no-path banner naming the active "
    "restrictions and suggesting a remedy. Nothing is raised, no partial path is invented, and the "
    "map keeps its other elements intact."
)
paper.paragraph(
    "The same behaviour was verified with two stronger perturbations executed from the interface: "
    "sealing every corridor incident to the destination, and sealing every corridor incident to the "
    "start. Both produce the no-path state within the same interaction cycle. Because the planner "
    "returns before reconstructing a path when the goal never enters the parent map, these cases "
    "cannot produce a KeyError in the reconstruction step."
)
paper.subheading("C. Constraint Validation")
paper.paragraph(
    "Reporting a route is not sufficient evidence of correctness, so the product independently "
    "validates every returned path. For each consecutive pair the validator re-queries the graph, "
    "discards blocked pairs, discards inaccessible edges when the flag is set, and fails the hop "
    "if no edge remains. The comparison table exposes the outcome of that check per algorithm. In "
    "the full suite across the five scenarios and three algorithms the validator reported no "
    "violations, while the single expected failure case was classified as a safe no-path report."
)
paper.paragraph(
    "Table III summarises the suite as the interface presents it: baseline cost, constrained cost, "
    "detour ratio and verdict. Detour ratios near unity indicate that the closure is bypassed with "
    "little penalty; a missing constrained cost marks a scenario in which the request is "
    "intentionally impossible."
)

stress_rows = []
for scenario in scenarios:
    identifier = scenario["id"]
    constraints = scenario.get("constraints") or {}
    outcome = results[identifier]["A*"]
    baseline = results["S1_normal_route"]["A*"]["cost"] if identifier in ("S1_normal_route", "S2_blocked_edge") else None
    if baseline is None:
        base_run = measure(graph, "A*", scenario["start"], scenario["end"], [], bool(constraints.get("accessible_only", False)))
        baseline = base_run["cost"]
    baseline_text = f"{baseline:.1f}" if baseline is not None else "-"
    if outcome["cost"] is None:
        verdict = "no path (expected)" if scenario["class"] == "failure" else "no path"
        stress_rows.append([identifier.split("_")[0], baseline_text, "-", "-", verdict])
    else:
        ratio = outcome["cost"] / baseline if baseline else 1.0
        stress_rows.append(
            [
                identifier.split("_")[0],
                baseline_text,
                f"{outcome['cost']:.1f}",
                f"{ratio:.2f}",
                "route recalculated" if ratio > 1.001 else "route unchanged",
            ]
        )

paper.table(
    "TABLE III",
    "Stress suite: baseline versus constrained cost and the reported verdict",
    ["Sc.", "Base", "Constr.", "Ratio", "Verdict"],
    stress_rows,
    [0.7, 0.9, 0.9, 0.8, 2.4],
    size=7.2,
)

paper.heading("VIII", "Discussion and Limitations")
paper.paragraph(
    "The measurements confirm the textbook ordering of effort. BFS expands the whole reachable "
    "component whenever the goal sits at the far side of the graph, because it has no notion of "
    "direction; its guarantee is about hop count, not about cost. GBFS is the cheapest in "
    "expansions and the most attractive for a live demo, but its route can be measurably longer, "
    "and in graphs with misleading geometry it can explore a large region before recovering. A* "
    "keeps the optimal cost while cutting expansion by a factor that depends on how informative the "
    "metric is, which is exactly the behaviour that makes the Euclidean heuristic worth its extra "
    "arithmetic."
)
paper.paragraph(
    "The product has deliberate limits. Routes are computed on a static schematic graph: crowd "
    "levels influence the travel-time weight but not the live state of a corridor, and there is no "
    "time-of-day model. Maintenance blocks are expressed as corridor pairs, so a user can close a "
    "corridor but not an entire building; a building closure is therefore simulated by sealing all "
    "of its incident corridors, which the isolation controls approximate. The interface is "
    "single-user and stateless across sessions, which is appropriate for a local demonstration "
    "product but would need persistence and authentication for deployment."
)
paper.paragraph(
    "A second limitation is methodological. Runtime is measured in-process on one machine and is "
    "reported as a median over 60 repetitions, so it reflects algorithmic effort far more than "
    "wall-clock reliability. The expansion counts and peak frontier sizes are the more portable "
    "indicators, and they are the ones the interface presents first."
)

paper.heading("IX", "Conclusion")
paper.paragraph(
    "The project delivers a working constraint-aware campus planner rather than a set of isolated "
    "search functions. Three algorithms implemented from first principles agree on the graph model "
    "and differ in exactly the ways the theory predicts: A* recovers optimal cost with "
    f"{expansion_ratio_s4:.1f}\u00d7 fewer expansions than BFS in the comparison scenario, GBFS "
    "minimises expansion at the price of optimality, and the Euclidean heuristic stays admissible "
    "throughout while the Manhattan estimate does not, because diagonal corridors break it. The "
    "interactive layer turns those differences into a product: landmarks, accessibility and "
    "maintenance blocks are the only inputs a user needs, and the same constraints drive the map, "
    "the metric table, the turn-by-turn directions and the stress suite."
)
paper.paragraph(
    "Future work would extend the state space rather than the algorithms: time-dependent corridor "
    "costs from crowd telemetry, multi-floor lift queues with capacity, and a small preferences "
    "layer that trades distance against crowding. The search core already accepts a blocked set and "
    "an accessibility flag on every call, so such extensions fit behind the existing interface "
    "without changing the product's contract."
)

paper.heading("", "References")
references = [
    "[1] S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed. Hoboken, NJ, USA: Pearson, 2020, pp. 63-102.",
    "[2] P. E. Hart, N. J. Nilsson, and B. Raphael, \u201cA formal basis for the heuristic determination of minimum cost paths,\u201d IEEE Trans. Syst. Sci. Cybern., vol. 4, no. 2, pp. 100-107, Jul. 1968.",
    "[3] J. Pearl, Heuristics: Intelligent Search Strategies for Computer Problem Solving. Reading, MA, USA: Addison-Wesley, 1984, pp. 33-72.",
    "[4] A. Dechter and J. Pearl, \u201cGeneralized best-first search strategies and the optimality of A*,\u201d J. ACM, vol. 32, no. 3, pp. 505-536, Jul. 1985.",
    "[5] S. Koenig and M. Likhachev, \u201cD* Lite,\u201d in Proc. AAAI Conf. Artif. Intell., Edmonton, AB, Canada, 2002, pp. 476-482.",
    "[6] Streamlit Inc., \u201cStreamlit documentation,\u201d 2024. [Online]. Available: https://docs.streamlit.io",
]
for reference in references:
    leading = 8.0 * 1.18 / 72.0
    prop = prop_for(8.0)
    indent_width = 0.30
    lines = paper.wrap(reference, COL_W - indent_width, prop)
    if paper.y - len(lines) * leading < MARGIN:
        paper.new_column()
    paper._draw_lines(lines, paper.x0() + indent_width, COL_W - indent_width, prop, 8.0, leading, justify=False)
    paper.y -= 0.02

paper.finish()
print(ROOT / "report.pdf")
