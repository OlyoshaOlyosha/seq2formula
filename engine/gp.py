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

import random
import time
from dataclasses import dataclass, field

from .fitness import Score, INVALID_FITNESS, score_candidate
from .grammar import (BINARY_OPS, MAX_INIT_DEPTH, crossover,
                      initial_population, point_mutation, shrink_mutation,
                      subtree_mutation)
from .nodes import Node, size, to_dict, to_text

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


@dataclass
class GPConfig:
    # Defaults tuned on the deception case (2^n vs the n^2 local optimum,
    # 6 seeds x 800 gens): pop 500 + mutation 0.25 solved 5/6; higher
    # mutation or bigger populations did not help further.
    population: int = 500
    max_depth: int = 6
    tournament: int = 3
    p_crossover: float = 0.85
    p_mutate: float = 0.25
    elitism: int = 2
    parsimony: float = 0.003
    max_generations: int = 1000
    seed: int | None = None
    allowed_ops: tuple[str, ...] | None = None  # None = all BINARY_OPS


@dataclass
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


@dataclass
class GenerationResult:
    gen: int
    best: Best
    new_best: bool          # fitness improved over the whole run
    solved_now: bool        # first exact hit in this generation
    solved: bool            # run has an exact formula (any past gen)
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
            "stats": self.stats,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class GPEngine:
    def __init__(self, seq: list[float], config: GPConfig | None = None):
        self.seq = [float(x) for x in seq]
        self.cfg = config or GPConfig()
        self.ops: tuple[str, ...] = (tuple(self.cfg.allowed_ops)
                                     if self.cfg.allowed_ops else BINARY_OPS)
        self.rng = random.Random(self.cfg.seed)
        self.generation = 0
        self.population: list[Node] = initial_population(
            self.rng, self.cfg.population, self.cfg.max_depth, self.ops)
        self.best_ever: Best | None = None
        self.shortest_exact: Best | None = None
        self.solved = False
        self.solved_gen: int | None = None
        self.curiosities: dict[tuple, dict] = {}

    # -- one generation -------------------------------------------------

    @property
    def done(self) -> bool:
        """True when the run has no reason to continue: out of budget, or
        solved and the shrink phase is over."""
        if self.generation >= self.cfg.max_generations:
            return True
        return self.solved and self.generation - self.solved_gen >= SHRINK_GENERATIONS

    def step(self) -> GenerationResult:
        started = time.perf_counter()
        # Once solved, the selection race is about shortness: a heavy
        # parsimony weight makes any size difference outweigh error noise.
        parsimony = SHRINK_PARSIMONY if self.solved else self.cfg.parsimony
        scored = [(node, score_candidate(node, self.seq, parsimony))
                  for node in self.population]
        # Stats come from the raw scores: the winner-take-all override
        # below would plunge the dashboard's mean fitness to -1.
        stats = self._stats(
            sorted(scored, key=lambda pair: pair[1].fitness, reverse=True))
        if self.solved:
            # Winner-take-all shrink: almost-correct trees must not outrank
            # exact ones, or selection drifts off the solution and the
            # shortening stalls. Only exact candidates compete on size.
            for _, sc in scored:
                if not sc.exact:
                    sc.fitness = INVALID_FITNESS
        scored.sort(key=lambda pair: pair[1].fitness, reverse=True)
        best_node, best_score = scored[0]

        new_best = (self.best_ever is None
                    or best_score.fitness > self.best_ever.score.fitness + 1e-12)
        if new_best:
            self.best_ever = Best(best_node, best_score)

        solved_now = best_score.exact and not self.solved
        if best_score.exact and not self.solved:
            self.solved = True
            self.solved_gen = self.generation
        # Scan the whole population: a size-3 gem can sit mid-table while
        # the fitness leader is still a size-9 exact monster.
        for node, sc in scored:
            if sc.exact and (self.shortest_exact is None
                             or sc.size < self.shortest_exact.score.size):
                self.shortest_exact = Best(node, sc)

        self._collect_curiosities(scored)
        top = []
        for node, sc in scored[:5]:
            top.append({
                "text": to_text(node),
                "ast": to_dict(node),
                "fitness": round(sc.fitness, 6),
                "size": sc.size,
                "error": None if sc.error == float("inf")
                         else round(sc.error, 8),
            })
        result = GenerationResult(
            gen=self.generation,
            best=Best(best_node, best_score),
            new_best=new_best,
            solved_now=solved_now,
            solved=self.solved,
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
            exact = sorted((pair for pair in scored if pair[1].exact),
                           key=lambda pair: pair[1].size)
            pool = exact if exact else scored
            elite = [node for node, _ in pool[:cfg.elitism]]
        else:
            elite = [node for node, _ in scored[:cfg.elitism]]
        offspring: list[Node] = []
        while len(elite) + len(offspring) < cfg.population:
            a = self._tournament(scored)
            if self.rng.random() < cfg.p_crossover:
                child = crossover(self.rng, a, self._tournament(scored),
                                  cfg.max_depth, self.ops)
            else:
                child = a
            if self.solved:
                # Shrink phase: hunt shorter exact forms on purpose.
                if self.rng.random() < 0.4:
                    child = shrink_mutation(self.rng, child)
                if self.rng.random() < cfg.p_mutate:
                    child = subtree_mutation(self.rng, child, cfg.max_depth, self.ops)
            else:
                if self.rng.random() < cfg.p_mutate:
                    child = subtree_mutation(self.rng, child, cfg.max_depth, self.ops)
                if self.rng.random() < 0.05:
                    child = point_mutation(self.rng, child, self.ops)
            offspring.append(child)
        return elite + offspring

    def _tournament(self, scored: list[tuple[Node, Score]]) -> Node:
        if self.solved:
            # During the shrink phase, breed only within the exact pool:
            # with two elites in 500 trees a blind tournament never picks
            # them, exact genes never spread, and shortening stalls.
            exact = [pair for pair in scored if pair[1].exact]
            if len(exact) >= 2:
                contenders = [exact[self.rng.randrange(len(exact))]
                              for _ in range(self.cfg.tournament)]
                winner = max(contenders, key=lambda pair: pair[1].fitness)
                return winner[0]
        contenders = [scored[self.rng.randrange(len(scored))]
                      for _ in range(self.cfg.tournament)]
        winner = max(contenders, key=lambda pair: pair[1].fitness)
        return winner[0]

    # -- statistics -----------------------------------------------------

    def _stats(self, scored: list[tuple[Node, Score]]) -> dict:
        scores = [s for _, s in scored]
        sizes = [s.size for s in scores]
        hist = [0] * SIZE_HIST_BINS
        for s in sizes:
            hist[min(s, SIZE_HIST_BINS) - 1] += 1
        valid = [s for s in scores if s.values is not None]
        diversity = (len({tuple(round(v, 4) for v in s.values) for s in valid})
                     / len(scored))
        return {
            "best_fitness": round(scores[0].fitness, 6),
            "mean_fitness": round(sum(s.fitness for s in scores) / len(scores), 6),
            "best_error": None if scores[0].error == float("inf")
                          else round(scores[0].error, 8),
            "mean_error": (round(sum(s.error for s in valid) / len(valid), 8)
                           if valid else None),
            "mean_size": round(sum(sizes) / len(sizes), 2),
            "best_size": scores[0].size,
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
