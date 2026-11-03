"""Tests for the curve-fitting engine (engine/curve.py)."""

import math
import random

import pytest

from engine.curve import (
    SOLVE_ERROR,
    CurveConfig,
    CurveGP,
    CurveTarget,
    _grid_of,
    _nearest_in_grid,
)
from engine.fitness import INVALID_FITNESS
from engine.nodes import BinOp, Const, Var

LINE = [(0.0, 0.0), (1.0, 1.0)]
DIV_ZERO = BinOp("/", Const(1.0), Const(0.0))


def test_target_validation() -> None:
    with pytest.raises(ValueError):
        CurveTarget([(0.5, 0.5)])
    with pytest.raises(ValueError):
        CurveTarget([(0.5, 0.5), (1.5, 0.1)])


def test_nearest_grid_matches_bruteforce() -> None:
    rng = random.Random(7)
    pts = [(rng.random(), rng.random()) for _ in range(40)]
    target = CurveTarget(pts)
    grid = _grid_of(pts)
    for _ in range(200):
        x, y = rng.random(), rng.random()
        expected = min(math.hypot(px - x, py - y) for px, py in pts)
        assert _nearest_in_grid(grid, pts, x, y) == pytest.approx(
            expected, abs=1e-12)
        assert target.nearest_mean([(x, y)]) == pytest.approx(expected)


def test_invalid_pair_scores_invalid() -> None:
    engine = CurveGP(LINE, CurveConfig(seed=1))
    bad = (DIV_ZERO, Var())
    score = engine._score(bad, 0.01)
    assert score.fitness == INVALID_FITNESS
    assert score.error == math.inf
    assert score.xs is None


def test_seeded_runs_are_deterministic() -> None:
    errs = []
    for _ in range(2):
        eng = CurveGP(LINE, CurveConfig(seed=42, population=80))
        run = []
        for _ in range(3):
            run.append(round(eng.step().best.score.error, 10))
        errs.append(run)
    assert errs[0] == errs[1]


def test_line_target_solves() -> None:
    eng = CurveGP(LINE, CurveConfig(seed=0, population=300, max_generations=300))
    best = 1.0
    for _ in range(300):
        best = eng.step().best.score.error
        if eng.solved:
            break
    assert eng.solved
    assert best < SOLVE_ERROR


def test_circle_target_converges() -> None:
    pts = [(0.5 + 0.5 * math.cos(2 * math.pi * i / 24),
            0.5 + 0.5 * math.sin(2 * math.pi * i / 24)) for i in range(24)]
    eng = CurveGP(pts, CurveConfig(seed=0, population=200, max_generations=150,
                                   allowed_funcs=("sin", "cos"),
                                   allowed_consts=("pi",)))
    start = eng.step().best.score.error
    best = start
    for _ in range(149):
        best = eng.step().best.score.error
    assert best < 0.15
    assert best < start


def test_payload_shape() -> None:
    eng = CurveGP(LINE, CurveConfig(seed=0, population=50))
    payload = eng.step().payload()
    assert payload["mode"] == "curve"
    for key in ("gen", "best", "top", "stats", "solved", "population"):
        assert key in payload
    assert len(payload["best"]["curve"]["xs"]) == 64
    assert len(payload["best"]["curve"]["ys"]) == 64
    assert payload["best"]["error"] is not None
    assert payload["stats"]["diversity"] > 0
