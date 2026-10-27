"""Settings sweep: find GP parameters that solve more puzzles, faster.

Usage:
  python scripts/sweep.py                     # full sweep, ~10-15 min
  python scripts/sweep.py --gens 300 --seeds 2
  python scripts/sweep.py --axes p_mutate     # only one axis
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Add the project root to sys.path so we can import `engine`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import GPConfig, GPEngine
from engine.grammar import BINARY_OPS
from engine.nodes import UNARY_FUNCS, special_value

BASIC_OPS = ("+", "-", "*", "/", "^")

AXES = {
    "parsimony": [0.001, 0.003, 0.01],
    "p_mutate": [0.15, 0.25, 0.4],
    "p_crossover": [0.7, 0.85, 1.0],
    "tournament": [2, 3, 5],
    "elitism": [1, 2, 4],
    "stagnation_window": [30, 60, 120],
}
DEFAULTS = {
    "parsimony": 0.003,
    "p_mutate": 0.25,
    "p_crossover": 0.85,
    "tournament": 3,
    "elitism": 2,
    "stagnation_window": 60,
}


def load_puzzles() -> list[dict]:
    with open(ROOT / "sequences.json", encoding="utf-8") as fh:
        return json.load(fh)["puzzles"]


def puzzle_grammar(puzzle: dict) -> dict:
    """Basic ops + exactly the features this puzzle needs."""
    requires = puzzle.get("requires", [])
    ops = tuple(dict.fromkeys(BASIC_OPS + tuple(
        r for r in requires if r in BINARY_OPS)))
    funcs = tuple(r for r in requires if r in UNARY_FUNCS)
    consts = tuple(r for r in requires if special_value(r) is not None)
    return {"allowed_ops": ops, "allowed_funcs": funcs,
            "allowed_consts": consts}


def run_config(puzzles: list[dict], overrides: dict, budget: int,
               seeds: list[int]) -> tuple[int, float, list[str]]:
    """Returns (solved_count, median generations, unsolved ids)."""
    gens_on_solve = []
    unsolved = []
    for puzzle in puzzles:
        solved = False
        for seed in seeds:
            config = GPConfig(seed=seed, population=300,
                              max_generations=budget, **puzzle_grammar(puzzle),
                              **overrides)
            engine = GPEngine([float(t) for t in puzzle["terms"]], config)
            while not engine.done:
                result = engine.step()
                if result.solved:
                    gens_on_solve.append(engine.solved_gen + 1)
                    solved = True
                    break
            if solved:
                break
        if not solved:
            unsolved.append(puzzle["id"])
    median = statistics.median(gens_on_solve) if gens_on_solve else float("nan")
    return len(puzzles) - len(unsolved), median, unsolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gens", type=int, default=300,
                        help="generation budget per puzzle run")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--axes", nargs="*", default=list(AXES),
                        help="which axes to sweep")
    parser.add_argument("--tiers", type=int, nargs="*", default=None,
                        help="restrict to tiers, e.g. --tiers 2 3")
    args = parser.parse_args()

    puzzles = load_puzzles()
    if args.tiers:
        puzzles = [p for p in puzzles if p["tier"] in args.tiers]
    seeds = list(range(1, args.seeds + 1))
    print(f"{len(puzzles)} puzzles, budget {args.gens} gens, "
          f"{args.seeds} seeds, pop 300\n")

    started = time.perf_counter()
    baseline = run_config(puzzles, {}, args.gens, seeds)
    print(f"BASELINE (current defaults): solved {baseline[0]}/{len(puzzles)}"
          f", median {baseline[1]:.0f} gens")

    for axis in args.axes:
        print(f"\n--- axis: {axis} ---")
        for value in AXES[axis]:
            overrides = dict(DEFAULTS) | {axis: value}
            count, median, unsolved = run_config(
                puzzles, overrides, args.gens, seeds)
            marker = " <- default" if value == DEFAULTS[axis] else ""
            print(f"  {axis}={value}: solved {count}/{len(puzzles)}"
                  f", median {median:.0f} gens{marker}")
            if unsolved and count < baseline[0]:
                print(f"    lost vs baseline, unsolved: {unsolved}")
        print(f"  ({time.perf_counter() - started:.0f}s elapsed)")

    print(f"\ntotal: {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()