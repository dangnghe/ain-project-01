import sys
import io
import json
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from graph import load_graph, load_scenarios
import student_core

graph = load_graph(ROOT / "data")
data   = load_scenarios(ROOT / "data")
scenarios = data["scenarios"]

HEURISTICS = {
    "euclidean": student_core.euclidean_heuristic,
    "manhattan": student_core.manhattan_heuristic,
}

SEP = "=" * 70

def fmt_path(path):
    if path is None:
        return "  ⚠ NO PATH FOUND"
    return "  " + " → ".join(path)

def run_astar(sc, heuristic_name="euclidean"):
    constraints = sc.get("constraints", {})
    blocked    = constraints.get("blocked_edges")
    accessible = constraints.get("accessible_only", False)
    h = HEURISTICS[heuristic_name]
    t0 = time.perf_counter()
    path, expanded = student_core.astar(
        graph, sc["start"], sc["end"], h,
        blocked_edges=blocked, accessible_only=accessible
    )
    elapsed = time.perf_counter() - t0
    cost = None
    if path:
        cost = sum(
            next(e.distance for e in graph.neighbors(path[i])
                 if (e.target == path[i+1] or e.source == path[i+1]))
            for i in range(len(path) - 1)
        )
    return path, expanded, elapsed, cost

for sc in scenarios:
    print(SEP)
    print(f"[{sc['id']}]  class={sc['class']}")
    print(f"  {sc['description']}")
    print(f"  Start: {sc['start']}  →  End: {sc['end']}")
    constraints = sc.get("constraints", {})
    if constraints:
        print(f"  Constraints: {json.dumps(constraints, ensure_ascii=False)}")
    print()

  
    if sc.get("heuristics"):
        print("  Heuristic comparison (A*):")
        for h_name in sc["heuristics"]:
            path, expanded, elapsed, cost = run_astar(sc, h_name)
            print(f"    [{h_name:10s}]  expanded={expanded:3d}  cost={cost:.2f}m  time={elapsed*1000:.3f}ms")
            print(fmt_path(path))
        print()
        continue
    path, expanded, elapsed, cost = run_astar(sc, "euclidean")
    if path is None:
        print("  Result: NO PATH FOUND ✓" if sc["class"] == "failure" else "  Result: ⚠ UNEXPECTED – no path found")
    else:
        print(f"  Result: path found  ({len(path)} nodes, cost={cost:.2f}m, expanded={expanded}, time={elapsed*1000:.3f}ms)")
    print(fmt_path(path))
    print()

print(SEP)
print("All scenarios completed.")
