"""Curve-fitting GP engine ("Кривые" mode).

Evolves a PAIR of formula trees x(t), y(t) per stroke: a drawing made of
K strokes gets K independent pairs, and the parametric curve of pair k
(64 samples of t in [0, 1]) must trace stroke k. A single continuous
pair can never cover disjoint strokes (a smiley), which is why the
genome carries one pair per stroke. A flat point list is accepted as a
single stroke, so one-stroke runs look exactly like the original API.

Fitness is the mean over strokes of a two-sided nearest-point distance:
every curve point should sit close to its stroke (shape fidelity) AND
every stroke point should sit close to the curve (coverage — without
this term a tiny curve hiding inside the shape would score well). Error
is normalized to [0, 1], so the "exact" threshold is a fraction of the
domain, not an epsilon: curve fitting has no exact hit in the sequence
sense.

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
    point_mutation,
    random_path,
    random_tree,
    replace_subtree,
    subtree_at,
    subtree_mutation,
)
from .nodes import (
    UNARY_FUNCS,
    BinOp,
    Const,
    Node,
    UnOp,
    Var,
    simplify,
    special_value,
    to_dict,
    to_text,
    weighted_size,
)

# Curve points sampled per candidate. 128, not 64: the coverage half of
# the fitness (target point -> nearest SAMPLE) has a granularity floor
# of ~half the sample spacing, and at 64 samples that floor (~0.02 of
# the unit square on a radius-0.4 circle) sat right AT the solve
# threshold — the perfect answer could not cross it. numpy batches keep
# 128 affordable.
CURVE_SAMPLES = 128
# "Solved" = mean two-sided distance below 2% of the unit square: visually
# the curve hugs the drawing. Not an epsilon — fitting never hits exact.
SOLVE_ERROR = 0.02
# Candidates per distance-tensor chunk in the batched fitness: the
# intermediate (chunk, 64, N_target) tensor stays ~10-30 MB.
DIST_CHUNK = 128
SIZE_HIST_BINS = 20
# Hard cap on strokes per drawing: init/scoring cost scales linearly
# with the stroke count, and a smiley needs 4.
MAX_STROKES = 6
# Curve targets live in the unit square, so the search needs fractional
# building blocks (stroke offsets like 0.25, affine maps like 0.1+0.8t)
# that the integer-oriented grammar can otherwise only assemble through
# division chains. They are injected into the specials pool, i.e. drawn
# like pi/e (SPECIAL_CONST_P per constant node) and cost weight 1
# (const_weight treats values < 1 as primitives).
CURVE_CONSTS = (0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1 / 3, 2 / 3)
# tau = 2*pi: with t in [0, 1] the full trig period needs 2*pi*t INSIDE
# sin/cos — a deep, rarely-assembled structure. tau*n is three nodes and
# makes the whole sin/cos family (circle, wave, star of Lissajous)
# directly reachable. Excluded from the gaussian tweak like pi/e.
CURVE_TAU = 2.0 * math.pi
# Generic oscillation seeds: with this probability a freshly generated
# tree component is "bias [+ amp *] sin/cos(k * t [+ phase])" instead of a
# plain random tree. This is a property of the GRAMMAR (curves are usually
# wavy; t runs over a normalized range), not a hint about any particular
# target — the same seeds help circle, wave, heart, star and a freehand
# doodle alike. Frequencies include tau/k so a full period fits t in
# [0, 1] exactly.
TRIG_SEED_RATE = 0.15
# tau twice: one full period over t in [0, 1] is the overwhelmingly
# common wavelength (circle, wave, heart, Lissajous), so it deserves
# extra draw weight; 2*tau covers two-period waves, tau/k are sub-periods.
TRIG_FREQS = (1.0, 2.0, 3.0, CURVE_TAU, CURVE_TAU, CURVE_TAU * 2,
              CURVE_TAU / 2, CURVE_TAU / 3)
# Fitness penalty weight against jagged curves: roughness (mean |second
# difference| / mean |step|, capped at 1) is subtracted from fitness,
# NEVER from error — the solve criterion stays untouched. Smooth analytic
# curves (circle at 128 samples) measure ~0.05, mod-jump monsters ~1, so
# the gap is up to ~0.05 fitness — enough to out-compete degenerate
# shortcuts without bending the error itself.
SMOOTH_W = 0.05
# Share of genomes born as a COHERENT oscillating block: one shared
# frequency k, x = bx + ax*sin(k*t), y = by + ay*cos|sin(k*t), with
# continuous ephemeral constants (classic GP practice — the discrete
# pool cannot hold an amplitude like 0.35). This is the natural
# parametric basis of the unit square (any Lissajous curve), not a hint
# about any target. Why coherent: fitness averages over the pair's
# components, so a great x next to a random y dies before crossover can
# marry the halves — only a block scores well AS A WHOLE. It need not
# be born exact: early on the monsters are still weak, a block just
# has to reach the elite, and the memetic pass walks its constants the
# rest of the way (amplitude precision is unreachable for crossover).
TRIG_BLOCK_RATE = 0.06


@dataclass(slots=True)
class CurveConfig:
    # Mirrors GPConfig; curve fitting differs only in max_depth (two trees
    # per candidate per stroke breed twice the size) and a much weaker
    # parsimony: a correct pair (circle) is 15+ nodes, and at the sequence
    # penalty 0.01 the 1-node diagonal beats it FOREVER on fitness — the
    # search never leaves degenerate shortcuts like x=t, y=t.
    population: int = 400
    max_depth: int = 5
    tournament: int = 3
    p_crossover: float = 1.0
    p_mutate: float = 0.25
    elitism: int = 2
    parsimony: float = 0.001
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
    # Memetic refinement: every generation the top-N valid unsolved
    # genomes get T random constant-nudge attempts each, keeping the best.
    # Crossover assembles the right SHAPE (a seeded sin/cos pair) but the
    # discrete constant pool cannot land an amplitude like 0.35 — the
    # last 2-4% of error live in the constants, and only local search
    # walks them down.
    memetic_elite: int = 3
    memetic_tries: int = 24


class CurveTarget:
    """One stroke's point cloud in the unit square, kept as a numpy
    array: the batched fitness measures every candidate against it."""

    def __init__(self, points: list[tuple[float, float]]):
        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) < 2:
            raise ValueError("a curve stroke needs at least 2 points")
        for x, y in pts:
            if not (math.isfinite(x) and math.isfinite(y)
                    and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("stroke points must lie in the unit square")
        self.points = pts
        self.array = np.asarray(pts, dtype=np.float64)

    def nearest_mean(self, points: list[tuple[float, float]]) -> float:
        """Mean distance from each given point to the closest stroke
        point (the "shape fidelity" half; kept for spot checks)."""
        arr = np.asarray(points, dtype=np.float64)
        diff = arr[:, None, :] - self.array[None, :, :]
        dist = np.sqrt((diff * diff).sum(-1))
        return float(dist.min(axis=1).mean())


def normalize_strokes(
        strokes: list[list[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    """Accept a list of strokes or one flat point list (back-compat);
    single-point strokes are padded to two identical points so a click
    stays a valid target. Enforces the stroke cap."""
    if not strokes:
        raise ValueError("a curve target needs at least one stroke")
    if isinstance(strokes[0][0], (int, float)):
        strokes = [strokes]  # flat point list -> single stroke
    if not (1 <= len(strokes) <= MAX_STROKES):
        raise ValueError(f"a drawing supports 1..{MAX_STROKES} strokes")
    out: list[list[tuple[float, float]]] = []
    for stroke in strokes:
        pts = [(float(x), float(y)) for x, y in stroke]
        if len(pts) == 1:
            pts = [pts[0], pts[0]]
        out.append(pts)
    return out


@dataclass(slots=True)
class CurveScore:
    fitness: float
    error: float          # mean two-sided distance in [0, 1]; inf = invalid
    exact: bool
    xs: list[list[float]] | None   # per stroke
    ys: list[list[float]] | None   # per stroke
    size: int
    vkey: bytes | None = None  # value-dedup key, precomputed at scoring


def _values_key(sc: CurveScore) -> tuple | bytes:
    """Dedup key from the sampled curves: comparisons stay by values, the
    same rule as the sequence engine (different trees, same curves = same
    answer). The key is built once per candidate by the scorer (a Python
    round-per-value pass costs ~30% of a generation otherwise)."""
    return sc.vkey if sc.vkey is not None else ()


@dataclass(slots=True)
class CurveBest:
    x_trees: tuple[Node, ...]   # one tree per stroke
    y_trees: tuple[Node, ...]
    score: CurveScore

    def entry(self, with_curve: bool) -> dict:
        out = {
            "x_text": [to_text(t) for t in self.x_trees],
            "x_ast": [to_dict(t) for t in self.x_trees],
            "y_text": [to_text(t) for t in self.y_trees],
            "y_ast": [to_dict(t) for t in self.y_trees],
            "size": self.score.size,
            "fitness": round(self.score.fitness, 6),
            "error": None if self.score.error == math.inf
                     else round(self.score.error, 8),
            "solved": self.score.exact,
        }
        if with_curve:
            out["curves"] = [
                {"xs": [round(v, 6) for v in xs], "ys": [round(v, 6) for v in ys]}
                for xs, ys in zip(self.score.xs, self.score.ys, strict=True)
            ]
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
    """GP over one pair of trees per stroke; the loop mirrors GPEngine
    step by step so the server can drive both engines through the same
    paced loop. Wiring is fixed: pair k always traces stroke k."""

    def __init__(self, strokes: list[list[tuple[float, float]]],
                 config: CurveConfig | None = None):
        strokes = normalize_strokes(strokes)
        self.targets = [CurveTarget(s) for s in strokes]
        self.n_strokes = len(self.targets)
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
        # Fractional constants are curve-intrinsic: they exist regardless
        # of the pi/e chips because no typical shape is reachable without
        # them. tau (2*pi) turns a full trig period into tau*n.
        self.specials = tuple(sorted(set(self.specials)
                                     | set(CURVE_CONSTS)
                                     | {CURVE_TAU}))
        self.rng = random.Random(self.cfg.seed)
        self.generation = 0
        # Ramped half-and-half per stroke over 2x trees, paired up: the
        # same init spread as the sequence engine, two trees per pair.
        # Some trees are generic trig seeds instead (TRIG_SEED_RATE).
        stroke_pairs: list[list[tuple[Node, Node]]] = []
        depths = list(range(2, self.cfg.max_depth + 1))
        for _ in self.targets:
            trees: list[Node] = []
            for i in range(self.cfg.population * 2):
                if self.rng.random() < TRIG_SEED_RATE:
                    trees.append(self._trig_seed())
                else:
                    trees.append(random_tree(
                        self.rng, depths[i % len(depths)],
                        "full" if i % 2 == 0 else "grow",
                        self.ops, self.funcs, self.specials))
            stroke_pairs.append(list(zip(trees[::2], trees[1::2], strict=True)))
        # genome = one pair per stroke
        self.population: list[tuple[tuple[Node, Node], ...]] = [
            tuple(sp[k] for sp in stroke_pairs)
            for k in range(self.cfg.population)
        ]
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

    def _trig_seed(self) -> Node:
        """Generic oscillating tree: bias [+ amp *] sin/cos(k*t [+ phase]).
        Gives crossover ready-made wave building blocks; frequency k is
        drawn so that a full period lands inside t in [0, 1]."""
        rng = self.rng
        inner: Node = BinOp("*", Const(rng.choice(TRIG_FREQS)), Var())
        if rng.random() < 0.3:
            inner = BinOp("+", inner, Const(rng.choice(self.specials)))
        tree: Node = UnOp(rng.choice(("sin", "cos")), inner)
        # Amplitude and bias from the fractional pool only: pi/e/tau as
        # an amplitude or offset is noise — useful seeds live in [0, 1].
        if rng.random() < 0.6:
            tree = BinOp("*", Const(rng.choice(CURVE_CONSTS)), tree)
        if rng.random() < 0.6:
            tree = BinOp("+", Const(rng.choice(CURVE_CONSTS)), tree)
        return tree

    def _random_tree(self, max_depth: int) -> Node:
        if self.rng.random() < TRIG_SEED_RATE:
            return self._trig_seed()
        return random_tree(self.rng, max_depth, "grow", self.ops,
                           self.funcs, self.specials)

    def _trig_block(self) -> tuple[Node, Node]:
        """Coherent oscillating pair with one shared frequency and
        ephemeral (continuous) constants — see TRIG_BLOCK_RATE."""
        rng = self.rng
        k = rng.choice(TRIG_FREQS)
        y_trig = "cos" if rng.randrange(2) == 0 else "sin"
        trees: list[Node] = []
        for trig, lo, hi in (("sin", 0.05, 0.55), (y_trig, 0.05, 0.55)):
            amp = rng.uniform(lo, hi)
            bias = rng.uniform(0.1, 0.9)
            tree: Node = UnOp(trig, BinOp("*", Const(k), Var()))
            if rng.random() < 0.7:
                tree = BinOp("*", Const(amp), tree)
            if rng.random() < 0.8:
                tree = BinOp("+", Const(bias), tree)
            trees.append(tree)
        return trees[0], trees[1]

    def _random_pair(self, max_depth: int) -> tuple[Node, Node]:
        if self.rng.random() < TRIG_BLOCK_RATE:
            return self._trig_block()
        return (self._random_tree(max_depth), self._random_tree(max_depth))

    def _random_genome(self, max_depth: int) -> tuple[tuple[Node, Node], ...]:
        return tuple(self._random_pair(max_depth) for _ in self.targets)

    def _genome_size(self, genome: tuple[tuple[Node, Node], ...]) -> int:
        return int(sum(weighted_size(x) + weighted_size(y)
                       for x, y in genome))

    def _sample(self, genome: tuple[tuple[Node, Node], ...]) -> tuple[
            list[list[float]] | None, list[list[float]] | None, int]:
        """Compile and sample every stroke's pair at the t-grid. Returns
        (None, None, size) if ANY stroke's pair hits a domain/overflow
        problem — a candidate must trace every stroke. Pure Python — each
        tree is a tiny distinct closure; the distance math is batched."""
        size = self._genome_size(genome)
        xs_all: list[list[float]] = []
        ys_all: list[list[float]] = []
        try:
            for x_tree, y_tree in genome:
                fx = compile_node(x_tree)
                fy = compile_node(y_tree)
                xs = [fx(t) for t in self._ts]
                ys = [fy(t) for t in self._ts]
                for v in xs + ys:
                    # A fractional power of a negative base returns
                    # complex, not an exception — check before isfinite.
                    if (isinstance(v, complex) or not math.isfinite(v)
                            or abs(v) > BIG):
                        return None, None, size
                xs_all.append(xs)
                ys_all.append(ys)
        except (ZeroDivisionError, OverflowError, ValueError, TypeError):
            return None, None, size
        return xs_all, ys_all, size

    def _batch_errors(self, curves: np.ndarray,
                      target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Two-sided nearest-point mean distances for a batch of curves
        against one stroke's target.

        curves: (M, CURVE_SAMPLES, 2). Squared distances expand via
        |c-t|^2 = |c|^2 + |t|^2 - 2 c·t so the inner loop is one GEMM
        instead of a broadcast subtract over a (chunk, 64, N, 2) tensor.
        One (chunk, 64, N) distance tensor serves both directions: min
        over the target axis = curve -> drawing fidelity, min over the
        sample axis = drawing -> curve coverage. One-sided fitness would
        reward a tiny curve hiding somewhere inside the shape.
        """
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
    def _roughness_batch(curves: np.ndarray) -> np.ndarray:
        """Jaggedness per curve (M, CURVE_SAMPLES, 2): mean absolute
        second difference divided by mean absolute step. ~0 for analytic
        curves, ~1 for mod-jump zigzags; capped at 1. Feeds fitness only,
        never error/solve."""
        if curves.shape[1] < 3:
            return np.zeros(curves.shape[0])
        d1 = curves[:, 1:] - curves[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        den = np.abs(d1).mean(axis=(1, 2))
        return np.minimum(np.abs(d2).mean(axis=(1, 2)) / (den + 1e-9), 1.0)

    @staticmethod
    def _vkey(curves: np.ndarray, index: int) -> bytes:
        return np.round(curves[index], 4).tobytes()

    def _score(self, genome: tuple[tuple[Node, Node], ...],
               parsimony: float) -> CurveScore:
        """Score a single genome (tests and spot checks); step() uses the
        batched path — same math, same values."""
        xs_all, ys_all, size = self._sample(genome)
        if xs_all is None:
            return CurveScore(INVALID_FITNESS, math.inf, False, None, None, size)
        error_sum = 0.0
        rough_sum = 0.0
        vkey = b""
        for k, target in enumerate(self.targets):
            curves = np.empty((1, CURVE_SAMPLES, 2))
            curves[0, :, 0] = xs_all[k]
            curves[0, :, 1] = ys_all[k]
            d_ct, d_tc = self._batch_errors(curves, target.array)
            error_sum += min(1.0, float(0.5 * (d_ct[0] + d_tc[0])))
            rough_sum += float(self._roughness_batch(curves)[0])
            vkey += np.round(curves, 4).tobytes()
        error = error_sum / self.n_strokes
        rough = rough_sum / self.n_strokes
        # Roughness only bends fitness (guides the search away from
        # jagged shortcuts); error and the solve threshold stay honest.
        fitness = (1.0 - error - SMOOTH_W * rough) - parsimony * size
        return CurveScore(fitness, error, error < SOLVE_ERROR, xs_all, ys_all,
                          size, vkey=vkey)

    # -- one generation ---------------------------------------------------

    @property
    def done(self) -> bool:
        return self.generation >= self.cfg.max_generations

    def step(self) -> CurveResult:
        started = time.perf_counter()
        # Phase 1: simplify, compile and sample — Python, per genome.
        genomes: list[tuple[tuple[Node, Node], ...]] = []
        scores: list[CurveScore] = []
        pending: list[int] = []   # indices of valid samples awaiting distances
        for genome in self.population:
            simp = tuple((simplify(x, self.ops), simplify(y, self.ops))
                         for x, y in genome)
            genomes.append(simp)
            xs_all, ys_all, size = self._sample(simp)
            if xs_all is None:
                scores.append(CurveScore(INVALID_FITNESS, math.inf, False,
                                         None, None, size))
            else:
                pending.append(len(scores))
                scores.append(CurveScore(0.0, 0.0, False, xs_all, ys_all, size))
        # Phase 2: all nearest-point distances in numpy batches, per
        # stroke; a candidate's error is the mean over its strokes.
        if pending:
            parsimony = self.cfg.parsimony
            errors = np.zeros(len(pending))
            rough = np.zeros(len(pending))
            vkeys: list[bytes] = [b""] * len(pending)
            for k, target in enumerate(self.targets):
                curves = np.empty((len(pending), CURVE_SAMPLES, 2))
                for row, i in enumerate(pending):
                    curves[row, :, 0] = scores[i].xs[k]
                    curves[row, :, 1] = scores[i].ys[k]
                d_ct, d_tc = self._batch_errors(curves, target.array)
                errors += np.minimum(1.0, 0.5 * (d_ct + d_tc))
                rough += self._roughness_batch(curves)
                # value-dedup key: the candidate's rounded curves across
                # ALL strokes, built once per generation
                rounded = np.round(curves, 4)
                for row in range(len(pending)):
                    vkeys[row] += rounded[row].tobytes()
            errors /= self.n_strokes
            rough /= self.n_strokes
            for row, i in enumerate(pending):
                sc = scores[i]
                error = float(errors[row])
                scores[i] = CurveScore(
                    (1.0 - error - SMOOTH_W * rough[row]) - parsimony * sc.size,
                    error, error < SOLVE_ERROR, sc.xs, sc.ys, sc.size,
                    vkey=vkeys[row])
        scored = list(zip(genomes, scores, strict=True))
        self.population = genomes

        stats = self._stats(scored)

        # Track the most accurate genome (lowest error), not the fitness
        # leader — the same no-regress rule as the sequence engine.
        true_best_error = math.inf
        true_best = scored[0]
        for genome, sc in scored:
            if sc.xs is not None and sc.error < true_best_error:
                true_best_error = sc.error
                true_best = (genome, sc)
        new_best = (self.best_ever is None
                    or true_best[1].error < self.best_ever.score.error - 1e-12)
        if new_best:
            self.best_ever = CurveBest(
                tuple(x for x, _ in true_best[0]),
                tuple(y for _, y in true_best[0]),
                true_best[1])

        # Fitness sharing: break clone armies of identical curves the
        # same way the sequence engine breaks identical outputs.
        keys = [_values_key(sc) if sc.xs is not None else () for _, sc in scored]
        counts = Counter(keys)
        for (genome, sc), key in zip(scored, keys, strict=True):
            if key and counts[key] > 1:
                sc.fitness -= (counts[key] - 1) * 0.001

        scored.sort(key=lambda item: item[1].fitness, reverse=True)
        self._memetic_refine(scored)

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
        seen: set[tuple | bytes] = set()
        for genome, sc in scored:
            if len(top) >= 5:
                break
            if sc.xs is None:
                continue
            key = _values_key(sc)
            if key in seen:
                continue
            seen.add(key)
            top.append({
                "x_text": [to_text(x) for x, _ in genome],
                "x_ast": [to_dict(x) for x, _ in genome],
                "y_text": [to_text(y) for _, y in genome],
                "y_ast": [to_dict(y) for _, y in genome],
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

    def _memetic_refine(
            self, scored: list[tuple[tuple[tuple[Node, Node], ...],
                                     CurveScore]]) -> None:
        """Local search on the fitness elite: T random single-constant
        nudges per elite genome, keep the variant only if its fitness
        (and error) improve. Writes replacements back into `scored` so
        _breed sees the refined genomes. Crossover assembles shapes;
        this walks the constants the pool cannot express exactly."""
        if self.cfg.memetic_elite <= 0:
            return
        for idx, (genome, sc) in enumerate(scored[:self.cfg.memetic_elite]):
            if sc.xs is None or sc.exact:
                continue
            best_g, best_sc = genome, sc
            for _ in range(self.cfg.memetic_tries):
                k = self.rng.randrange(self.n_strokes)
                x, y = best_g[k]
                i = self.rng.randrange(2)
                tree = self._tweak_const(x if i == 0 else y)
                if tree is None:
                    continue
                child = list(best_g)
                child[k] = (tree, y) if i == 0 else (x, tree)
                child_sc = self._score(tuple(child), self.cfg.parsimony)
                if (child_sc.fitness > best_sc.fitness
                        and child_sc.error < best_sc.error):
                    best_g, best_sc = tuple(child), child_sc
            if best_g is not genome:
                scored[idx] = (best_g, best_sc)

    def _breed(self, scored: list[tuple[tuple[tuple[Node, Node], ...],
                                        CurveScore]]
               ) -> list[tuple[tuple[Node, Node], ...]]:
        cfg = self.cfg
        elite = [genome for genome, _ in scored[:cfg.elitism]]
        offspring: list[tuple[tuple[Node, Node], ...]] = []
        # Random immigrants keep baseline diversity alive, as in gp.py.
        immigrants = int(cfg.population * cfg.random_immigrants)
        for _ in range(immigrants):
            if len(elite) + len(offspring) < cfg.population:
                offspring.append(self._random_genome(3))
        while len(elite) + len(offspring) < cfg.population:
            if self._shock_now and self.rng.random() < cfg.shock_fraction:
                # Shock trees stay shallow (depth 3): they incur almost
                # no parsimony penalty and survive selection better.
                offspring.append(self._random_genome(3))
                continue
            a = self._tournament(scored)
            if self.rng.random() < cfg.p_crossover:
                child = self._pair_crossover(a, self._tournament(scored))
            else:
                child = a
            if self.rng.random() < cfg.p_mutate:
                child = self._genome_mutate(child)
            offspring.append(child)
        return elite + offspring

    def _pair_crossover(self, a: tuple[tuple[Node, Node], ...],
                        b: tuple[tuple[Node, Node], ...]
                        ) -> tuple[tuple[Node, Node], ...]:
        # Recombine ONE randomly chosen component of ONE randomly chosen
        # stroke: everything else passes through untouched, so good
        # halves spread like genes.
        k = self.rng.randrange(self.n_strokes)
        x, y = a[k]
        bx, by = b[k]
        if self.rng.randrange(2) == 0:
            x = crossover(self.rng, x, bx, self.cfg.max_depth,
                          self.ops, self.funcs, self.specials)
        else:
            y = crossover(self.rng, y, by, self.cfg.max_depth,
                          self.ops, self.funcs, self.specials)
        child = list(a)
        child[k] = (x, y)
        return tuple(child)

    def _tweak_const(self, tree: Node) -> Node | None:
        """Gaussian nudge of one random constant, or None when the picked
        node is not a tunable constant. The discrete constant pool cannot
        fine-tune a 0.667 into the 0.75 a stroke actually needs — this is
        the classic ephemeral-constant mutation, curve edition."""
        path = random_path(self.rng, tree)
        node = subtree_at(tree, path)
        if not isinstance(node, Const):
            return None
        if node.value in (math.pi, math.e, CURVE_TAU):
            return None  # special constants are semantic, leave them be
        v = max(-3.0, min(3.0, node.value + self.rng.gauss(0, 0.05)))
        return replace_subtree(tree, path, Const(v))

    def _genome_mutate(self, genome: tuple[tuple[Node, Node], ...]
                       ) -> tuple[tuple[Node, Node], ...]:
        k = self.rng.randrange(self.n_strokes)
        x, y = genome[k]
        i = self.rng.randrange(2)
        tree = x if i == 0 else y
        tweaked = False
        # ~30% of mutations nudge a constant instead of restructuring
        if self.rng.random() < 0.3:
            tweaked_tree = self._tweak_const(tree)
            if tweaked_tree is not None:
                tree = tweaked_tree
                tweaked = True
        if not tweaked:
            if self.rng.random() < 0.05:
                tree = point_mutation(self.rng, tree, self.ops, self.funcs,
                                      self.specials)
            else:
                tree = subtree_mutation(self.rng, tree, self.cfg.max_depth,
                                        self.ops, self.funcs, self.specials)
        child = list(genome)
        child[k] = (tree, y) if i == 0 else (x, tree)
        return tuple(child)

    def _tournament(self, scored: list[tuple[tuple[tuple[Node, Node], ...],
                                             CurveScore]]
                    ) -> tuple[tuple[Node, Node], ...]:
        n = len(scored)
        winner, win_fit = scored[self.rng.randrange(n)]
        win_fit = win_fit.fitness
        for _ in range(self.cfg.tournament - 1):
            genome, sc = scored[self.rng.randrange(n)]
            if sc.fitness > win_fit:
                winner, win_fit = genome, sc.fitness
        return winner

    # -- statistics -------------------------------------------------------

    def _stats(self, scored: list[tuple[tuple[tuple[Node, Node], ...],
                                         CurveScore]]) -> dict:
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
