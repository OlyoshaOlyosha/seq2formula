"""Genetic programming engine (stage 1).

Classic GP: population of formula trees, tournament selection, subtree
crossover, subtree/point mutation, elitism. One step() = one generation,
so the server can pace the search (pause / step / generations per second)
and stream a result per generation to the dashboard.

After the first exact hit the engine keeps evolving: with parsimony in
the fitness the population keeps shrinking the formula, which is how the
hall of fame finds "the shortest correct answer". The server decides
when to stop.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from .fitness import INVALID_FITNESS, Score, score_candidate
from .grammar import (
    BINARY_OPS,
    crossover,
    initial_population,
    point_mutation,
    random_tree,
    shrink_mutation,
    subtree_mutation,
)
from .nodes import UNARY_FUNCS, Node, special_value, to_dict, to_text, simplify

# Curiosity gallery: "almost correct" (mean error below) but comically
# overbuilt (size at least) formulas, deduped by the values they produce.
CURIOSITY_MAX_ERROR = 0.2
CURIOSITY_MIN_SIZE = 8
CURIOSITY_CAP = 200

# After the first exact hit the run keeps evolving for this many
# generations, with a much stronger parsimony weight: among equally
# exact formulas the fitness race is purely about size, which is how
# the hall of fame finds "the shortest correct answer".
SHRINK_GENERATIONS = 120
SHRINK_PARSIMONY = 0.02

# Size histogram bins: 1..19 exact sizes, 20+ lumped together.
SIZE_HIST_BINS = 20

# Anti-stagnation ("shock"): when the best fitness has not improved for
# this many generations (and the run is not solved yet), a share of the
# population is replaced with fresh random trees — the usual GP failure
# is a degenerate population grinding smaller-and-smaller wrong formulas.
# 0 disables the mechanism entirely.
#
# Adaptive: the window is the WAIT before a shock. Each shock lasts
# `shock_duration` generations — a single-generation shot never sticks,
# fresh randoms get out-selected before they can breed. If a shock fails
# to produce a new best, the next wait grows by `stagnation_factor`
# (30 → 45 → 67 → 100 → ... capped at `stagnation_max`). Any new best
# resets the wait back to the base.
STAGNATION_WINDOW = 30
SHOCK_FRACTION = 0.25
SHOCK_DURATION = 10
STAGNATION_MAX = 120
STAGNATION_FACTOR = 1.2


@dataclass(slots=True)
class GPConfig:
    # Defaults tuned on the deception case (2^n vs the n^2 local optimum,
    # 6 seeds x 800 gens): pop 500 + mutation 0.25 solved 5/6; higher
    # mutation or bigger populations did not help further.
    population: int = 500
    max_depth: int = 6
    tournament: int = 3
    # Crossover on every child (mutations apply on top): the sweep over
    # 41 puzzles x 4 seeds solved one more puzzle than 0.85 with no losses.
    p_crossover: float = 1.0
    p_mutate: float = 0.25
    elitism: int = 2
    parsimony: float = 0.003
    max_generations: int = 10000
    seed: int | None = None
    allowed_ops: tuple[str, ...] | None = None    # None = all BINARY_OPS
    # v2 features default OFF so a plain GPConfig() searches exactly the
    # v1 space (tuned convergence); None explicitly means "all of them".
    allowed_funcs: tuple[str, ...] | None = ()
    allowed_consts: tuple[str, ...] | None = ()
    # shrink-phase knobs, exposed to the settings tab
    shrink_generations: int = SHRINK_GENERATIONS
    shrink_parsimony: float = SHRINK_PARSIMONY
    # anti-stagnation: base wait before a shock (0 = off entirely)
    stagnation_window: int = STAGNATION_WINDOW
    shock_fraction: float = SHOCK_FRACTION
    # how many generations each shock lasts, and adaptive escalation
    # of the wait between shocks
    shock_duration: int = SHOCK_DURATION
    stagnation_max: int = STAGNATION_MAX
    stagnation_factor: float = STAGNATION_FACTOR
    # fraction of each generation replaced by random trees to prevent
    # the population from collapsing into identical clones
    random_immigrants: float = 0.02


@dataclass(slots=True)
class Best:
    node: Node
    score: Score

    def payload(self) -> dict:
        return {
            "text": to_text(self.node),
            "ast": to_dict(self.node),
            "size": self.score.size,
            "fitness": round(self.score.fitness, 6),
            "error": None if self.score.error == float("inf") else round(self.score.error, 8),
            "exact": self.score.exact,
            "values": None if self.score.values is None
                      else [round(v, 6) for v in self.score.values],
            "errors": None if self.score.errors is None
                      else [round(e, 6) for e in self.score.errors],
        }


@dataclass(slots=True)
class GenerationResult:
    gen: int
    best: Best
    new_best: bool          # fitness improved over the whole run
    solved_now: bool        # first exact hit in this generation
    solved: bool            # run has an exact formula (any past gen)
    shock: bool             # stagnation shock fired this generation
    shock_window: int = 0   # current wait threshold (for the event feed)
    top: list[dict] = field(default_factory=list)  # leaderboard, best first
    stats: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0
    population: int = 0

    def payload(self) -> dict:
        return {
            "type": "gen",
            "gen": self.gen,
            "population": self.population,
            "best": self.best.payload(),
            "top": self.top,
            "new_best": self.new_best,
            "solved_now": self.solved_now,
            "solved": self.solved,
            "shock": self.shock,
            "shock_window": self.shock_window,
            "stats": self.stats,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class GPEngine:
    def __init__(self, seq: list[float], config: GPConfig | None = None):
        self.seq = [float(x) for x in seq]
        self.cfg = config or GPConfig()
        self.ops: tuple[str, ...] = (tuple(self.cfg.allowed_ops)
                                     if self.cfg.allowed_ops is not None
                                     else BINARY_OPS)
        # () (explicitly empty) means "functions disabled" — only None
        # means "all of them"; an `if not` check would confuse the two.
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
        self.population: list[Node] = initial_population(
            self.rng, self.cfg.population, self.cfg.max_depth, self.ops,
            self.funcs, self.specials)
        self.best_ever: Best | None = None
        self.shortest_exact: Best | None = None
        self.solved = False
        self.solved_gen: int | None = None
        self._exact_pool: list[tuple[Node, Score]] = []
        self.stagnation = 0   # generations since the last new best
        self.shocks = 0       # shocks fired over the whole run
        self._shock_now = False
        # adaptive shock state: current wait threshold (grows on each
        # useless shock, resets on any new best) and remaining shock gens
        self.current_window = self.cfg.stagnation_window
        self.shock_remaining = 0
        self.curiosities: dict[tuple, dict] = {}

    # -- one generation -------------------------------------------------

    @property
    def done(self) -> bool:
        """True when the run has no reason to continue: out of budget, or
        solved and the shrink phase is over."""
        if self.generation >= self.cfg.max_generations:
            return True
        return (self.solved
                and self.generation - self.solved_gen >= self.cfg.shrink_generations)

    def step(self) -> GenerationResult:
        started = time.perf_counter()
        # Once solved, the selection race is about shortness: a heavy
        # parsimony weight makes any size difference outweigh error noise.
        parsimony = (self.cfg.shrink_parsimony if self.solved
                     else self.cfg.parsimony)
        scored = []
        for node in self.population:
            simp_node = simplify(node, self.ops)
            scored.append((simp_node, score_candidate(simp_node, self.seq, parsimony)))
        self.population = [n for n, _ in scored]

        # Stats come from the raw scores: the winner-take-all override
        # below would plunge the dashboard's mean fitness to -1.
        stats = self._stats(scored)
        
        if self.solved:
            # Winner-take-all shrink: almost-correct trees must not outrank
            # exact ones, or selection drifts off the solution and the
            # shortening stalls. Only exact candidates compete on size.
            for _, sc in scored:
                if not sc.exact:
                    sc.fitness = INVALID_FITNESS
                    
        # Find the most accurate formula (lowest error). The UI must show 
        # the most precise formula, not just the one with the best fitness 
        # (which might be shorter but less accurate). Tracking the minimum 
        # error also prevents the displayed error from jumping around when 
        # parsimony penalizes a highly accurate but large formula.
        true_best_error = math.inf
        true_best_node = scored[0][0]
        true_best_score = scored[0][1]
        for node, sc in scored:
            if sc.values is not None and sc.error < true_best_error:
                true_best_error = sc.error
                true_best_node = node
                true_best_score = sc
                
        new_best = (self.best_ever is None
                    or true_best_score.error < self.best_ever.score.error - 1e-12)
        if new_best:
            self.best_ever = Best(true_best_node, true_best_score)

        # Fitness Sharing: penalize clones AFTER determining new_best.
        # If 400 out of 500 formulas produce the exact same output (like
        # the constant "1"), they form an unbeatable "clone army". We
        # subtract 0.001 per duplicate. A formula with 400 clones loses
        # 0.4 fitness, dropping below a diverse formula with a slightly
        # worse raw score.
        if not self.solved:
            from collections import Counter
            keys = [tuple(round(v, 4) for v in sc.values) if sc.values else ()
                    for _, sc in scored]
            counts = Counter(keys)
            for (node, sc), key in zip(scored, keys):
                if key and counts[key] > 1:
                    sc.fitness -= (counts[key] - 1) * 0.001
                    
        # Sort exactly once, after all fitness modifications
        scored.sort(key=lambda pair: pair[1].fitness, reverse=True)
        self._exact_pool = [pair for pair in scored if pair[1].exact]
        # Anti-stagnation: adaptive shock. Any new best resets the wait
        # to the base; a shock lasts `shock_duration` generations (a
        # single generation never sticks — fresh randoms get out-selected
        # before they can breed), and a useless shock escalates the next
        # wait by `stagnation_factor` up to `stagnation_max`.
        if new_best:
            self.stagnation = 0
            self.current_window = self.cfg.stagnation_window
            self.shock_remaining = 0   # cancel an ongoing shock
        elif self.shock_remaining == 0:
            self.stagnation += 1
        # (during a shock the wait counter stays frozen)
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

        # Solve detection scans the WHOLE population, not just the fitness
        # leader: a big-but-exact formula can legitimately lose on fitness
        # to a shorter near-miss (1 - error - parsimony*size), and it must
        # still trigger the solve + shrink phase. Same scan as the hall of
        # fame below.
        solved_now = (not self.solved) and any(sc.exact for _, sc in scored)
        if solved_now:
            self.solved = True
            self.solved_gen = self.generation
        # Scan the whole population: a size-3 gem can sit mid-table while
        # the fitness leader is still a size-9 exact monster.
        for node, sc in scored:
            if sc.exact and (self.shortest_exact is None
                             or sc.size < self.shortest_exact.score.size):
                self.shortest_exact = Best(node, sc)

        self._collect_curiosities(scored)
        # Top-10 by fitness, skipping invalid / shrink-overridden (-1)
        # entries and deduplicating by values so the leaderboard shows
        # 10 *distinct* formulas, not 10 copies of "1".
        top = []
        seen_values = set()
        for node, sc in scored:
            if len(top) >= 10:
                break
            if sc.fitness == INVALID_FITNESS and not self.solved:
                continue
            if self.solved and not sc.exact:
                continue  # during shrink, only show exact formulas
            if sc.values is not None:
                val_key = tuple(round(v, 4) for v in sc.values)
                if val_key in seen_values:
                    continue
                seen_values.add(val_key)
            top.append({
                "text": to_text(node),
                "ast": to_dict(node),
                "fitness": round(sc.fitness, 6),
                "size": sc.size,
                "error": None if sc.error == float("inf")
                         else round(sc.error, 8),
                "values": None if sc.values is None
                          else [round(v, 6) for v in sc.values],
            })
        # The UI must show the historical best (best_ever), otherwise
        # the displayed error jumps around when Fitness Sharing penalizes
        # clones and a worse formula temporarily wins the generation.
        display_best = self.best_ever
        if self.solved and self.shortest_exact:
            display_best = self.shortest_exact

        result = GenerationResult(
            gen=self.generation,
            best=display_best,
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

    # -- breeding -------------------------------------------------------

    def _breed(self, scored: list[tuple[Node, Score]]) -> list[Node]:
        cfg = self.cfg
        if self.solved:
            # Elites must stay exact: with shrink parsimony a short
            # almost-correct tree could outscore a long exact one and
            # drag the population back off the solution.
            exact = sorted(self._exact_pool, key=lambda pair: pair[1].size)
            pool = exact if exact else scored
            elite = [node for node, _ in pool[:cfg.elitism]]
        else:
            elite = [node for node, _ in scored[:cfg.elitism]]
        offspring: list[Node] = []
        # Random immigrants: a small, constant trickle of fresh trees
        # keeps baseline diversity from collapsing into clones.
        immigrants = int(cfg.population * cfg.random_immigrants)
        for _ in range(immigrants):
            if len(elite) + len(offspring) < cfg.population:
                offspring.append(random_tree(
                    self.rng, 3, "grow", self.ops, self.funcs, self.specials))
        
        while len(elite) + len(offspring) < cfg.population:
            if self._shock_now and self.rng.random() < cfg.shock_fraction:
                # Stagnation shock: shallow random trees (depth 3) survive
                # selection better than deep ones because they incur almost
                # no parsimony penalty and are less likely to be invalid.
                offspring.append(random_tree(
                    self.rng, 3, "grow", self.ops, self.funcs, self.specials))
                continue
            a = self._tournament(scored)
            if self.rng.random() < cfg.p_crossover:
                child = crossover(self.rng, a, self._tournament(scored),
                                  cfg.max_depth, self.ops, self.funcs,
                                  self.specials)
            else:
                child = a
            if self.solved:
                # Shrink phase: hunt shorter exact forms on purpose.
                if self.rng.random() < 0.4:
                    child = shrink_mutation(self.rng, child, self.specials)
                if self.rng.random() < cfg.p_mutate:
                    child = subtree_mutation(self.rng, child, cfg.max_depth,
                                             self.ops, self.funcs, self.specials)
            else:
                if self.rng.random() < cfg.p_mutate:
                    child = subtree_mutation(self.rng, child, cfg.max_depth,
                                             self.ops, self.funcs, self.specials)
                if self.rng.random() < 0.05:
                    child = point_mutation(self.rng, child, self.ops,
                                           self.funcs, self.specials)
            offspring.append(child)
        return elite + offspring

    def _tournament(self, scored: list[tuple[Node, Score]]) -> Node:
        randrange = self.rng.randrange
        if self.solved:
            # During the shrink phase, breed only within the exact pool:
            # with two elites in 500 trees a blind tournament never picks
            # them, exact genes never spread, and shortening stalls.
            # (scored is fitness-sorted, so a plain scan finds winners.)
            exact = self._exact_pool
            if len(exact) >= 2:
                n = len(exact)
                winner, win_fit = exact[randrange(n)]
                win_fit = win_fit.fitness
                for _ in range(self.cfg.tournament - 1):
                    node, sc = exact[randrange(n)]
                    if sc.fitness > win_fit:
                        winner, win_fit = node, sc.fitness
                return winner
        n = len(scored)
        winner, win_fit = scored[randrange(n)]
        win_fit = win_fit.fitness
        for _ in range(self.cfg.tournament - 1):
            node, sc = scored[randrange(n)]
            if sc.fitness > win_fit:
                winner, win_fit = node, sc.fitness
        return winner

    # -- statistics -----------------------------------------------------

    def _stats(self, scored: list[tuple[Node, Score]]) -> dict:
        scores = [s for _, s in scored]
        sizes = [s.size for s in scores]
        hist = [0] * SIZE_HIST_BINS
        for s in sizes:
            hist[min(s, SIZE_HIST_BINS) - 1] += 1
        valid = [s for s in scores if s.values is not None]
        # Denominator is the VALID count, not the total: fresh random
        # shock trees are often invalid (div-zero / domain / overflow),
        # and including them in the denominator capped the KPI at ~50%
        # even when every valid formula was unique.
        diversity = (len({tuple(round(v, 4) for v in s.values) for s in valid})
                     / max(1, len(valid)))
                     
        best_score = max(scores, key=lambda s: s.fitness)
        
        return {
            "best_fitness": round(best_score.fitness, 6),
            "mean_fitness": round(sum(s.fitness for s in scores) / len(scores), 6),
            "best_error": None if best_score.error == float("inf")
                          else round(best_score.error, 8),
            "mean_error": (round(sum(s.error for s in valid) / len(valid), 8)
                           if valid else None),
            "mean_size": round(sum(sizes) / len(sizes), 2),
            "best_size": best_score.size,
            "size_hist": hist,
            "diversity": round(diversity, 4),
            "valid_frac": round(len(valid) / len(scores), 4),
        }

    def _collect_curiosities(self, scored: list[tuple[Node, Score]]) -> None:
        for node, sc in scored:
            if (sc.values is None or sc.exact
                    or sc.error >= CURIOSITY_MAX_ERROR
                    or sc.size < CURIOSITY_MIN_SIZE):
                continue
            key = tuple(round(v, 6) for v in sc.values)
            current = self.curiosities.get(key)
            if current is None or sc.size > current["best"].score.size:
                self.curiosities[key] = {
                    "best": Best(node, sc),
                    "gen": self.generation,
                }
        if len(self.curiosities) > CURIOSITY_CAP:
            # Keep the most absurd (largest) ones.
            kept = sorted(self.curiosities.values(),
                          key=lambda e: e["best"].score.size, reverse=True)
            self.curiosities = {tuple(round(v, 6) for v in e["best"].score.values): e
                                for e in kept[:CURIOSITY_CAP]}

    def curiosity_payloads(self, limit: int = 30) -> list[dict]:
        entries = sorted(self.curiosities.values(),
                         key=lambda e: e["best"].score.size, reverse=True)[:limit]
        return [{"formula": e["best"].payload(), "gen": e["gen"]} for e in entries]
