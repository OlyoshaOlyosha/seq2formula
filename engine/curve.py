"""Curve-fitting GP engine ("Кривые" mode).

Evolves a PAIR of formula trees x(t), y(t) per candidate; the parametric
curve sampled at t in [0, 1] must trace a target point cloud drawn in the
unit square. The GP machinery (grammar, crossover, mutations, repairs,
shocks) is reused from the sequence engine unchanged — only the genome
and the fitness differ.

Fitness is a two-sided nearest-point distance: every curve point should
sit close to the drawing (shape fidelity) AND every drawn point should
sit close to the curve (coverage — without this term a tiny circle
somewhere inside the shape would score well). Error is normalized to
[0, 1], so the "exact" threshold is a fraction of the domain, not an
epsilon: curve fitting has no exact hit in the sequence sense.

The nearest-point distances are computed by numpy in population-sized
batches (one distance tensor per chunk serves both directions); without
this the two-sided metric over hundreds of candidates x 64 samples is
too slow in pure Python (numpy agreed as a dependency 2026-11-06).
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .evaluator import BIG, compile_node
from .fitness import INVALID_FITNESS
from .grammar import (
    BINARY_OPS,
    crossover,
    initial_population,
    point_mutation,
    random_tree,
    subtree_mutation,
)
from .nodes import (
    UNARY_FUNCS,
    Node,
    simplify,
    special_value,
    to_dict,
    to_text,
    weighted_size,
)

# Curve points sampled per candidate: enough to trace any single stroke
# smoothly, small enough that 400 candidates per generation stay fast.
CURVE_SAMPLES = 64
# "Solved" = mean two-sided distance below 2% of the unit square: visually
# the curve hugs the drawing. Not an epsilon — fitting never hits exact.
SOLVE_ERROR = 0.02
# Candidates per distance-tensor chunk in the batched fitness: the
# intermediate (chunk, 64, N_target) tensor stays ~10-30 MB.
DIST_CHUNK = 128
SIZE_HIST_BINS = 20


@dataclass(slots=True)
class CurveConfig:
    # Mirrors GPConfig; curve fitting differs only in max_depth (two trees
    # per candidate breed twice the size) and a stronger parsimony default
    # (an unpenalized pair drifts into wiggly monsters fast).
    population: int = 400
    max_depth: int = 5
    tournament: int = 3
    p_crossover: float = 1.0
    p_mutate: float = 0.25
    elitism: int = 2
    parsimony: float = 0.01
    max_generations: int = 10000
    seed: int | None = None
    allowed_ops: tuple[str, ...] | None = None
    allowed_funcs: tuple[str, ...] | None = ()
    allowed_consts: tuple[str, ...] | None = ()
    stagnation_window: int = 30
    shock_fraction: float = 0.25
    shock_duration: int = 10
    stagnation_max: int = 120
    stagnation_factor: float = 1.2
    random_immigrants: float = 0.02


class CurveTarget:
    """Target point cloud in the unit square, kept as a numpy array:
    the batched fitness measures every candidate against it at once."""

    def __init__(self, points: list[tuple[float, float]]):
        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) < 2:
            raise ValueError("a curve target needs at least 2 points")
        for x, y in pts:
            if not (math.isfinite(x) and math.isfinite(y)
                    and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("target points must lie in the unit square")
        self.points = pts
        self.array = np.asarray(pts, dtype=np.float64)

    def nearest_mean(self, points: list[tuple[float, float]]) -> float:
        """Mean distance from each given point to the closest target
        point (the "shape fidelity" half; kept for spot checks)."""
        arr = np.asarray(points, dtype=np.float64)
        diff = arr[:, None, :] - self.array[None, :, :]
        dist = np.sqrt((diff * diff).sum(-1))
        return float(dist.min(axis=1).mean())


@dataclass(slots=True)
class CurveScore:
    fitness: float
    error: float          # mean two-sided distance in [0, 1]; inf = invalid
    exact: bool
    xs: list[float] | None
    ys: list[float] | None
    size: int
    vkey: bytes | None = None  # value-dedup key, precomputed at scoring


def _values_key(sc: CurveScore) -> tuple | bytes:
    """Dedup key from the sampled curve: comparisons stay by values, the
    same rule as the sequence engine (different trees, same curve = same
    answer). The key is built once per candidate by the scorer (a Python
    round-per-value pass costs ~30% of a generation otherwise)."""
    return sc.vkey if sc.vkey is not None else ()


@dataclass(slots=True)
class CurveBest:
    x_tree: Node
    y_tree: Node
    score: CurveScore

    def entry(self, with_curve: bool) -> dict:
        out = {
            "x_text": to_text(self.x_tree),
            "x_ast": to_dict(self.x_tree),
            "y_text": to_text(self.y_tree),
            "y_ast": to_dict(self.y_tree),
            "size": self.score.size,
            "fitness": round(self.score.fitness, 6),
            "error": None if self.score.error == math.inf
                     else round(self.score.error, 8),
            "solved": self.score.exact,
        }
        if with_curve:
            out["curve"] = {
                "xs": [round(v, 6) for v in self.score.xs],
                "ys": [round(v, 6) for v in self.score.ys],
            }
        return out


@dataclass(slots=True)
class CurveResult:
    gen: int
    best: CurveBest
    new_best: bool
    solved_now: bool
    solved: bool
    shock: bool
    shock_window: int = 0
    top: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0
    population: int = 0

    def payload(self) -> dict:
        return {
            "type": "gen",
            "mode": "curve",
            "gen": self.gen,
            "population": self.population,
            "best": self.best.entry(with_curve=True),
            "top": self.top,
            "new_best": self.new_best,
            "solved_now": self.solved_now,
            "solved": self.solved,
            "shock": self.shock,
            "shock_window": self.shock_window,
            "stats": self.stats,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class CurveGP:
    """GP over pairs of trees; the loop mirrors GPEngine step by step so
    the server can drive both engines through the same paced loop."""

    def __init__(self, points: list[tuple[float, float]],
                 config: CurveConfig | None = None):
        self.target = CurveTarget(points)
        self.cfg = config or CurveConfig()
        self.ops: tuple[str, ...] = (tuple(self.cfg.allowed_ops)
                                     if self.cfg.allowed_ops is not None
                                     else BINARY_OPS)
        # () (explicitly empty) means "functions disabled" — only None
        # means "all of them"; the same convention as GPEngine.
        self.funcs: tuple[str, ...] = (tuple(self.cfg.allowed_funcs)
                                       if self.cfg.allowed_funcs is not None
                                       else UNARY_FUNCS)
        self.specials: tuple[float, ...] = tuple(
            v for v in (special_value(token)
                        for token in (self.cfg.allowed_consts
                                      if self.cfg.allowed_consts is not None
                                      else ("pi", "e")))
            if v is not None)
        self.rng = random.Random(self.cfg.seed)
        self.generation = 0
        # Ramped half-and-half over 2x trees, paired up: the same init
        # spread as the sequence engine, two trees per candidate.
        trees = initial_population(self.rng, self.cfg.population * 2,
                                   self.cfg.max_depth, self.ops,
                                   self.funcs, self.specials)
        self.population: list[tuple[Node, Node]] = list(
            zip(trees[::2], trees[1::2], strict=True))
        self.best_ever: CurveBest | None = None
        self.solved = False
        self.solved_gen: int | None = None
        self.stagnation = 0   # generations since the last new best
        self.shocks = 0       # shocks fired over the whole run
        self._shock_now = False
        self.current_window = self.cfg.stagnation_window
        self.shock_remaining = 0
        self._ts = [i / (CURVE_SAMPLES - 1) for i in range(CURVE_SAMPLES)]

    # -- genome helpers ---------------------------------------------------

    def _random_pair(self, max_depth: int) -> tuple[Node, Node]:
        return (random_tree(self.rng, max_depth, "grow", self.ops,
                            self.funcs, self.specials),
                random_tree(self.rng, max_depth, "grow", self.ops,
                            self.funcs, self.specials))

    def _sample(self, pair: tuple[Node, Node]) -> tuple[
            list[float] | None, list[float] | None, int]:
        """Compile both trees and sample them at the t-grid. Returns
        (None, None, size) on any domain/overflow problem. Pure Python —
        each tree is a tiny distinct closure, the per-tree loop is
        unavoidable; the expensive distance math below is batched."""
        x_tree, y_tree = pair
        size = int(weighted_size(x_tree) + weighted_size(y_tree))
        try:
            fx = compile_node(x_tree)
            fy = compile_node(y_tree)
            xs = [fx(t) for t in self._ts]
            ys = [fy(t) for t in self._ts]
            for v in xs + ys:
                # A fractional power of a negative base returns complex,
                # not an exception — type-check before isfinite.
                if isinstance(v, complex) or not math.isfinite(v) or abs(v) > BIG:
                    return None, None, size
        except (ZeroDivisionError, OverflowError, ValueError, TypeError):
            return None, None, size
        return xs, ys, size

    def _batch_errors(self, curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Two-sided nearest-point mean distances for a batch of curves.

        curves: (M, CURVE_SAMPLES, 2). Squared distances expand via
        |c-t|^2 = |c|^2 + |t|^2 - 2 c·t so the inner loop is one GEMM
        instead of a broadcast subtract over a (chunk, 64, N, 2) tensor.
        One (chunk, 64, N) distance tensor serves both directions: min
        over the target axis = curve -> drawing fidelity, min over the
        sample axis = drawing -> curve coverage. One-sided fitness would
        reward a tiny curve hiding somewhere inside the shape.
        """
        target = self.target.array
        t_sq = (target * target).sum(axis=1)
        m = curves.shape[0]
        d_ct = np.empty(m)
        d_tc = np.empty(m)
        for lo in range(0, m, DIST_CHUNK):
            hi = min(lo + DIST_CHUNK, m)
            flat = curves[lo:hi].reshape(-1, 2)
            d2 = ((flat * flat).sum(axis=1)[:, None] + t_sq[None, :]
                  - 2.0 * (flat @ target.T))
            np.maximum(d2, 0.0, out=d2)
            dist3 = np.sqrt(d2).reshape(hi - lo, CURVE_SAMPLES, -1)
            d_ct[lo:hi] = dist3.min(axis=2).mean(axis=1)
            d_tc[lo:hi] = dist3.min(axis=1).mean(axis=1)
        return d_ct, d_tc

    @staticmethod
    def _vkey(curves: np.ndarray, index: int) -> bytes:
        return np.round(curves[index], 4).tobytes()

    def _score(self, pair: tuple[Node, Node], parsimony: float) -> CurveScore:
        """Score a single pair (tests and spot checks); step() uses the
        batched path — same math, same values."""
        xs, ys, size = self._sample(pair)
        if xs is None:
            return CurveScore(INVALID_FITNESS, math.inf, False, None, None, size)
        curves = np.empty((1, CURVE_SAMPLES, 2))
        curves[0, :, 0] = xs
        curves[0, :, 1] = ys
        d_ct, d_tc = self._batch_errors(curves)
        error = min(1.0, float(0.5 * (d_ct[0] + d_tc[0])))
        fitness = (1.0 - error) - parsimony * size
        return CurveScore(fitness, error, error < SOLVE_ERROR, xs, ys, size,
                          vkey=self._vkey(curves, 0))

    # -- one generation ---------------------------------------------------

    @property
    def done(self) -> bool:
        return self.generation >= self.cfg.max_generations

    def step(self) -> CurveResult:
        started = time.perf_counter()
        # Phase 1: simplify, compile and sample — Python, per pair.
        pairs: list[tuple[Node, Node]] = []
        scores: list[CurveScore] = []
        pending: list[int] = []   # indices of valid samples awaiting distances
        for pair in self.population:
            simp = (simplify(pair[0], self.ops), simplify(pair[1], self.ops))
            pairs.append(simp)
            xs, ys, size = self._sample(simp)
            if xs is None:
                scores.append(CurveScore(INVALID_FITNESS, math.inf, False,
                                         None, None, size))
            else:
                pending.append(len(scores))
                scores.append(CurveScore(0.0, 0.0, False, xs, ys, size))
        # Phase 2: all nearest-point distances in one numpy batch.
        if pending:
            curves = np.empty((len(pending), CURVE_SAMPLES, 2))
            for k, i in enumerate(pending):
                curves[k, :, 0] = scores[i].xs
                curves[k, :, 1] = scores[i].ys
            d_ct, d_tc = self._batch_errors(curves)
            parsimony = self.cfg.parsimony
            for k, i in enumerate(pending):
                sc = scores[i]
                error = min(1.0, 0.5 * (float(d_ct[k]) + float(d_tc[k])))
                scores[i] = CurveScore(
                    (1.0 - error) - parsimony * sc.size, error,
                    error < SOLVE_ERROR, sc.xs, sc.ys, sc.size,
                    vkey=self._vkey(curves, k))
        scored = list(zip(pairs, scores, strict=True))
        self.population = pairs

        stats = self._stats(scored)

        # Track the most accurate pair (lowest error), not the fitness
        # leader — the same no-regress rule as the sequence engine.
        true_best_error = math.inf
        true_best = scored[0]
        for pair, sc in scored:
            if sc.xs is not None and sc.error < true_best_error:
                true_best_error = sc.error
                true_best = (pair, sc)
        new_best = (self.best_ever is None
                    or true_best[1].error < self.best_ever.score.error - 1e-12)
        if new_best:
            self.best_ever = CurveBest(true_best[0][0], true_best[0][1],
                                       true_best[1])

        # Fitness sharing: break clone armies of identical curves the
        # same way the sequence engine breaks identical outputs.
        keys = [_values_key(sc) if sc.xs is not None else () for _, sc in scored]
        counts = Counter(keys)
        for (pair, sc), key in zip(scored, keys, strict=True):
            if key and counts[key] > 1:
                sc.fitness -= (counts[key] - 1) * 0.001

        scored.sort(key=lambda item: item[1].fitness, reverse=True)

        # Adaptive anti-stagnation shock, same semantics as GPEngine.
        if new_best:
            self.stagnation = 0
            self.current_window = self.cfg.stagnation_window
            self.shock_remaining = 0
        elif self.shock_remaining == 0:
            self.stagnation += 1
        if (not self.solved and self.cfg.stagnation_window > 0
                and self.shock_remaining == 0
                and self.stagnation >= self.current_window):
            self.shock_remaining = self.cfg.shock_duration
            self.shocks += 1
            self.stagnation = 0
            self.current_window = min(
                self.cfg.stagnation_max,
                int(self.current_window * self.cfg.stagnation_factor))
        self._shock_now = self.shock_remaining > 0
        if self._shock_now:
            self.shock_remaining -= 1

        solved_now = (not self.solved) and any(sc.exact for _, sc in scored)
        if solved_now:
            self.solved = True
            self.solved_gen = self.generation

        top = []
        seen: set[tuple] = set()
        for pair, sc in scored:
            if len(top) >= 5:
                break
            if sc.xs is None:
                continue
            key = _values_key(sc)
            if key in seen:
                continue
            seen.add(key)
            top.append({
                "x_text": to_text(pair[0]), "x_ast": to_dict(pair[0]),
                "y_text": to_text(pair[1]), "y_ast": to_dict(pair[1]),
                "size": sc.size,
                "fitness": round(sc.fitness, 6),
                "error": None if sc.error == math.inf else round(sc.error, 8),
                "solved": sc.exact,
            })

        result = CurveResult(
            gen=self.generation,
            best=self.best_ever,
            new_best=new_best,
            solved_now=solved_now,
            solved=self.solved,
            shock=self._shock_now,
            shock_window=self.current_window if self._shock_now else 0,
            top=top,
            stats=stats,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            population=self.cfg.population,
        )
        self.generation += 1
        self.population = self._breed(scored)
        return result

    # -- breeding ---------------------------------------------------------

    def _breed(self, scored: list[tuple[tuple[Node, Node], CurveScore]]
               ) -> list[tuple[Node, Node]]:
        cfg = self.cfg
        elite = [pair for pair, _ in scored[:cfg.elitism]]
        offspring: list[tuple[Node, Node]] = []
        # Random immigrants keep baseline diversity alive, as in gp.py.
        immigrants = int(cfg.population * cfg.random_immigrants)
        for _ in range(immigrants):
            if len(elite) + len(offspring) < cfg.population:
                offspring.append(self._random_pair(3))
        while len(elite) + len(offspring) < cfg.population:
            if self._shock_now and self.rng.random() < cfg.shock_fraction:
                # Shock trees stay shallow (depth 3): they incur almost
                # no parsimony penalty and survive selection better.
                offspring.append(self._random_pair(3))
                continue
            a = self._tournament(scored)
            if self.rng.random() < cfg.p_crossover:
                child = self._pair_crossover(a, self._tournament(scored))
            else:
                child = a
            if self.rng.random() < cfg.p_mutate:
                child = self._pair_mutate(child)
            offspring.append(child)
        return elite + offspring

    def _pair_crossover(self, a: tuple[Node, Node],
                        b: tuple[Node, Node]) -> tuple[Node, Node]:
        # Recombine ONE randomly chosen component: the other passes
        # through untouched, so good halves spread like genes.
        i = self.rng.randrange(2)
        child = list(a)
        child[i] = crossover(self.rng, a[i], b[i], self.cfg.max_depth,
                             self.ops, self.funcs, self.specials)
        return (child[0], child[1])

    def _pair_mutate(self, pair: tuple[Node, Node]) -> tuple[Node, Node]:
        i = self.rng.randrange(2)
        tree = pair[i]
        if self.rng.random() < 0.05:
            tree = point_mutation(self.rng, tree, self.ops, self.funcs,
                                  self.specials)
        else:
            tree = subtree_mutation(self.rng, tree, self.cfg.max_depth,
                                    self.ops, self.funcs, self.specials)
        child = list(pair)
        child[i] = tree
        return (child[0], child[1])

    def _tournament(self, scored: list[tuple[tuple[Node, Node], CurveScore]]
                    ) -> tuple[Node, Node]:
        n = len(scored)
        winner, win_fit = scored[self.rng.randrange(n)]
        win_fit = win_fit.fitness
        for _ in range(self.cfg.tournament - 1):
            pair, sc = scored[self.rng.randrange(n)]
            if sc.fitness > win_fit:
                winner, win_fit = pair, sc.fitness
        return winner

    # -- statistics -------------------------------------------------------

    def _stats(self, scored: list[tuple[tuple[Node, Node], CurveScore]]) -> dict:
        scores = [s for _, s in scored]
        sizes = [s.size for s in scores]
        hist = [0] * SIZE_HIST_BINS
        for s in sizes:
            hist[min(s, SIZE_HIST_BINS) - 1] += 1
        valid = [s for s in scores if s.xs is not None]
        # Denominator is the VALID count: fresh shock pairs are often
        # invalid, and including them capped diversity at ~50%.
        diversity = (len({_values_key(s) for s in valid})
                     / max(1, len(valid)))
        best_score = max(scores, key=lambda s: s.fitness)
        return {
            "best_fitness": round(best_score.fitness, 6),
            "mean_fitness": round(sum(s.fitness for s in scores) / len(scores), 6),
            "best_error": None if best_score.error == math.inf
                          else round(best_score.error, 8),
            "mean_error": (round(sum(s.error for s in valid) / len(valid), 8)
                           if valid else None),
            "mean_size": round(sum(sizes) / len(sizes), 2),
            "best_size": best_score.size,
            "size_hist": hist,
            "diversity": round(diversity, 4),
            "valid_frac": round(len(valid) / len(scores), 4),
        }
