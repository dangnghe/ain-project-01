

Interactive campus wayfinding product: a Streamlit app that plans step-free,
maintenance-aware routes with BFS, greedy best-first search and A*, plus the
IEEE-format technical report (`report.pdf`).


Requires **Python 3.11+** and Git. No API keys, database or network access are
needed; everything runs locally on the committed data.

```bash
git clone <your-repo-url> project-01-campus-route
cd project-01-campus-route

python3.11 -m venv .venv
source .venv/bin/activate           
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (default <http://localhost:8501>); open it in a
browser.

Verification and regeneration:

bash
pytest tests -q                          
python scripts/build_report.py          
(cd data && python test_scenarios.py)    
(cd data && sha256sum -c SHA256SUMS)     


`python scripts/build_report.py` writes `report.pdf` in the repository root and
uses only `matplotlib`, `pandas` and the search core; it needs no extra packages.


```text
project-01-campus-route/
├── app.py                   
├── report.pdf                
├── data/                    
├── starter/
│   ├── graph.py              
│   ├── student_core.py       
│   ├── visuals.py            
│   └── app.py               
├── scripts/
│   ├── generate_campus.py   
│   └── build_report.py      
├── tests/                  
├── requirements.txt
└── .gitattributes            



`pytest tests` yields one failure on purpose:
`test_student_core_stubs_raise_not_implemented` verifies that the AI core is
*unimplemented*, which stops being true once `starter/student_core.py` is filled
in. All data-integrity, loader and checksum tests pass.



Navigating a modern multi-building campus requires routing algorithms that handle real-world layout constraints, accessibility needs, and dynamic path obstructions. In this project students formulate campus navigation as a state-space graph search problem: given a weighted graph of campus locations and connections, an automated planner must find optimal or near-optimal routes between arbitrary landmarks while accommodating temporarily blocked pathways and step-free accessibility requirements.

The core challenge is to implement and evaluate classical graph search algorithms from first principles—comparing uninformed baselines against informed heuristic search—and to analyse trade-offs in search effort, memory footprint, and path cost optimality across a realistic set of navigation scenarios.




The project graph models a generic multi-building campus with **54 nodes** and **75 bidirectional edges** annotated with distance, travel time, crowd level, and accessibility attributes.


```text
project-01-campus-route/
├── data/
│   ├── nodes.csv          
│   ├── edges.csv          
│   ├── scenarios.json    
│   ├── campus_map.png     
│   ├── SHA256SUMS         
│   └── README.md         
├── starter/
│   ├── graph.py           
│   ├── student_core.py    
│   └── app.py            
├── tests/
│   └── test_project_01_sanity.py
└── requirements.txt





`starter/graph.py` provides graph loading, neighbour queries, and property lookups. `starter/app.py` renders the campus schematic and path overlays. Students implement all AI logic in `starter/student_core.py`.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run starter/app.py
```



- **Implement**: `bfs`, `greedy`, `astar`, `euclidean_heuristic`, and `manhattan_heuristic` in `starter/student_core.py` from scratch. External graph search solvers (e.g., NetworkX search) are prohibited.
- **Experiment**: Run all scenarios in `data/scenarios.json` and compare BFS, GBFS, and A* on (i) nodes expanded, (ii) peak frontier size, (iii) path cost, and (iv) runtime. Formally prove heuristic admissibility ($h(n) \le h^*(n)$) and consistency ($h(n) \le c(n,a,n') + h(n')$).
- **Deliverables**: Completed `student_core.py`, evaluation notebook or script, `presentation.pdf`.
- **Grading Criteria**: Problem formulation and algorithmic correctness (15 pts); experimental rigour and heuristic analysis (10 pts); oral defence and live demo (15 pts). **Total: 40 pts.**



- **Interactive Application**: Streamlit app where users select start and end landmarks, toggle accessible-only routing, dynamically block paths to simulate maintenance, and compare algorithm routes side-by-side.
- **Stress Scenarios**: Demonstrate path recalculation after primary corridors are blocked; correctly detect and report no-path cases without crashing.
- **Deliverables**: Fully operational local web application, GitHub repository, `report.pdf` (IEEE format).
- **Grading Criteria**: Product functionality, UX, and UI integration (25 pts); technical report depth and architectural documentation (15 pts); live defence under stress scenarios (10 pts). **Total: 50 pts.**



```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

`app.py` at the repository root is the interactive product (landmark selection,
accessibility toggle, maintenance blocks, side-by-side algorithm comparison,
stress suite and benchmark). `starter/app.py` stays the untouched skeleton it
was shipped as, and `report.pdf` is the IEEE-format technical report, rebuildable
with `python scripts/build_report.py`.

`pytest tests` reports one expected failure:
`test_student_core_stubs_raise_not_implemented` asserts that the AI core is
still unimplemented, which is no longer true once `starter/student_core.py` is
implemented.


- [1] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Hoboken, NJ, USA: Pearson, 2020, pp. 63–102.
- [2] P. E. Hart, N. J. Nilsson, and B. Raphael, "A formal basis for the heuristic determination of minimum cost paths," *IEEE Transactions on Systems Science and Cybernetics*, vol. 4, no. 2, pp. 100–107, Jul. 1968, doi: 10.1109/TSSC.1968.300136.
- [3] J. Pearl, *Heuristics: Intelligent Search Strategies for Computer Problem Solving*. Reading, MA, USA: Addison-Wesley, 1984, pp. 33–72.



Released under the MIT License — see [LICENSE](LICENSE).
Copyright (c) 2026 Bùi Tá Minh Đăng.


`.github/workflows/ci.yml` runs on every push and pull request with Python 3.11. It
lints the authored code with ruff (`E9,F` rules), verifies `data/SHA256SUMS`, and
runs `pytest tests` with the obsolete starter guard test deselected (see the note
in section 3.3).

