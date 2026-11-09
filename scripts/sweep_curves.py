"""Headless benchmark for the curve engine: run every dashboard preset
shape through CurveGP and report final error / solved / speed.

Usage: python -m scripts.sweep_curves [--pop 400] [--gens 1000]
Prints one line per shape; exit code 0 always. Used to compare engine
tuning batches (baseline vs changes) without touching the UI.
"""

import argparse
import math
import time

from engine.curve import CurveConfig, CurveGP


def along_path(verts: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Same as the dashboard's alongPath(): polyline sampled by length."""
    segs: list[float] = []
    for (x0, y0), (x1, y1) in zip(verts, verts[1:], strict=False):
        segs.append(math.hypot(x1 - x0, y1 - y0))
    total = sum(segs)
    pts: list[tuple[float, float]] = []
    acc = 0.0
    step = total / (n - 1)
    want = 0.0
    for (x0, y0), (x1, y1) in zip(verts, verts[1:], strict=False):
        seg = math.hypot(x1 - x0, y1 - y0)
        while acc + seg >= want and len(pts) < n:
            f = (want - acc) / seg if seg else 0.0
            pts.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
            want += step
        acc += seg
    while len(pts) < n:
        pts.append(pts[-1])
    return pts


def presets() -> dict[str, list[tuple[float, float]]]:
    tau = 2 * math.pi
    return {
        "line": along_path([(0.12, 0.5), (0.88, 0.5)], 30),
        "angle": along_path([(0.2, 0.15), (0.2, 0.85), (0.85, 0.85)], 48),
        "circle": [(0.5 + 0.35 * math.cos(-math.pi / 2 + i / 47 * tau),
                    0.5 + 0.35 * math.sin(-math.pi / 2 + i / 47 * tau))
                   for i in range(48)],
        "wave": [(0.1 + 0.8 * i / 59, 0.5 + 0.25 * math.sin(4 * math.pi * i / 59))
                 for i in range(60)],
        "zigzag": along_path([(0.1, 0.25), (0.3, 0.75), (0.5, 0.25),
                              (0.7, 0.75), (0.9, 0.25)], 60),
        "spiral": [(0.5 + (0.04 + 0.32 * t) * math.cos(-math.pi / 2 + t * 2.5 * tau),
                    0.5 + (0.04 + 0.32 * t) * math.sin(-math.pi / 2 + t * 2.5 * tau))
                   for t in (i / 89 for i in range(90))],
        "star": along_path(
            [(0.5 + (0.38 if k % 2 == 0 else 0.16) * math.cos(-math.pi / 2 + k * math.pi / 5),
              0.5 + (0.38 if k % 2 == 0 else 0.16) * math.sin(-math.pi / 2 + k * math.pi / 5))
             for k in range(10)] + [(0.5, 0.12)], 70),
        "heart": [(0.5 + 0.32 * math.sin(th) ** 3,
                   0.35 - 0.02 * (13 * math.cos(th) - 5 * math.cos(2 * th)
                                  - 2 * math.cos(3 * th) - math.cos(4 * th)))
                  for th in (i / 59 * tau for i in range(60))],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=400)
    ap.add_argument("--gens", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    print(f"pop={args.pop} gens<={args.gens} seed={args.seed} {args.label}")
    for name, pts in presets().items():
        eng = CurveGP([pts], CurveConfig(seed=args.seed, population=args.pop,
                                         max_generations=args.gens))
        t0 = time.perf_counter()
        err = 1.0
        solved_gen = None
        for _ in range(args.gens):
            res = eng.step()
            err = res.best.score.error
            if eng.solved:
                solved_gen = eng.generation
                break
        secs = time.perf_counter() - t0
        gps = (solved_gen or args.gens) / secs if secs else 0.0
        print(f"{name:8s} err={err * 100:6.2f}% "
              f"{'SOLVED@' + str(solved_gen) if solved_gen else 'no     '} "
              f"{gps:5.1f} gen/s", flush=True)


if __name__ == "__main__":
    main()
