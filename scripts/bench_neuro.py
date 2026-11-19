"""Headless benchmark for the RL neuro engine: run library puzzles
through NeuroEngine in parallel worker processes and report final error /
solved generation / wall time.

Usage: python -m scripts.bench_neuro [--tier 2] [--seeds 0 1 2]
       [--gens 500] [--workers 8]
One line per (puzzle, seed); a per-puzzle summary at the end. Used to
compare engine tuning batches (baseline vs changes) without the UI.
"""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from engine.neuro import NeuroConfig, NeuroEngine


def load_puzzles(tier: int | None) -> list[dict]:
    root = Path(__file__).resolve().parent.parent
    with open(root / "engine" / "sequences.json", encoding="utf-8") as f:
        data = json.load(f)
    puzzles = data["puzzles"] if isinstance(data, dict) else data
    return [p for p in puzzles if tier is None or p.get("tier") == tier]


def run_one(args: tuple[str, list[float], int, int]) -> dict:
    """Worker body: must stay top-level for multiprocessing on Windows."""
    name, terms, seed, gens = args
    eng = NeuroEngine(terms, NeuroConfig(seed=seed))
    t0 = time.perf_counter()
    solved = None
    for g in range(gens):
        eng.step()
        if eng.solved:
            solved = g + 1
            break
    elapsed = time.perf_counter() - t0
    best = eng.best_ever.score.error if eng.best_ever else float("nan")
    return {"name": name, "seed": seed, "solved": solved,
            "error": best, "seconds": elapsed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=(0,))
    ap.add_argument("--gens", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    jobs = [(p["name"], p["terms"], s, a.gens)
            for p in load_puzzles(a.tier) for s in a.seeds]
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(run_one, jobs))
    total = time.perf_counter() - t0

    for r in results:
        mark = f"solved@{r['solved']}" if r["solved"] else "no"
        print(f"{r['name']} seed{r['seed']}: {mark} "
              f"err={r['error'] * 100:.2f}% {r['seconds']:.0f}s", flush=True)

    by_puzzle: dict[str, list[dict]] = {}
    for r in results:
        by_puzzle.setdefault(r["name"], []).append(r)
    solved_n = 0
    print("-" * 40)
    for name, rows in by_puzzle.items():
        n = sum(1 for r in rows if r["solved"])
        solved_n += n
        errs = ", ".join(f"s{r['seed']}:{r['error'] * 100:.2f}%" for r in rows)
        print(f"{name}: {n}/{len(rows)} seeds solved ({errs})")
    print(f"TOTAL: {solved_n}/{len(results)} solved | wall {total:.0f}s")


if __name__ == "__main__":
    main()
