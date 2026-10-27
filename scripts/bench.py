"""Engine speed benchmark — the reference point for optimization work.

Measures pure engine throughput (no server, no browser, single process),
so numbers are comparable across machines/commits regardless of how many
dashboard tabs are open. Run:  python scripts/bench.py
"""

import sys
import time
from pathlib import Path

# Add the project root to sys.path so we can import `engine`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import GPConfig, GPEngine
from engine.nodes import UNARY_FUNCS, size

V2_FUNCS = UNARY_FUNCS
V2_SPECIALS = (3.141592653589793, 2.718281828459045)

SCENARIOS = [
    ("t1-linear  2n        (v1 ops)", [2, 4, 6, 8, 10, 12, 14], 200,
     dict(allowed_ops=("+", "-", "*", "/"))),
    ("t2-power   2^n       (v1 ops)", [2, 4, 8, 16, 32, 64, 128], 300,
     dict(allowed_ops=("+", "-", "*", "/", "^"))),
    ("t3-mixed   n^2-n+1    (v1 ops)", [1, 3, 7, 13, 21, 31, 43], 300,
     dict(allowed_ops=("+", "-", "*", "/", "^"))),
    ("t4-factorial n!       (v2 fact)", [1, 2, 6, 24, 120, 720], 200,
     dict(allowed_ops=("+", "-", "*", "/"), allowed_funcs=("fact",))),
    ("heavy      all v2 features      ", [1, 2, 6, 24, 120, 720], 200,
     dict(allowed_ops=("+", "-", "*", "/", "^", "mod"),
          allowed_funcs=V2_FUNCS, allowed_consts=V2_SPECIALS)),
]


def run(name: str, terms, gens: int, config_kw: dict) -> None:
    engine = GPEngine([float(t) for t in terms],
                      GPConfig(seed=42, **config_kw))
    per_gen = []
    for _ in range(gens):
        started = time.perf_counter()
        engine.step()
        per_gen.append((time.perf_counter() - started) * 1000.0)
    sizes = [size(node) for node in engine.population]
    head = sum(per_gen[:50]) / min(50, len(per_gen))
    tail = sum(per_gen[-50:]) / min(50, len(per_gen))
    mean = sum(per_gen) / len(per_gen)
    print(f"{name} | {mean:6.2f} ms/gen | {1000 / mean:6.1f} gen/s "
          f"| first50 {head:6.2f} vs last50 {tail:6.2f} ms "
          f"| mean size {sum(sizes) / len(sizes):.1f}")


if __name__ == "__main__":
    print(f"{'scenario':36s} |  speed                    | drift (bloat)")
    for scenario in SCENARIOS:
        run(*scenario)