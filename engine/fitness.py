"""Fitness: continuous closeness reward plus a parsimony penalty.

The user's formulation, kept exactly:
- the candidate is applied to n = 1..len(seq); the closer the values to
  the target, the higher the reward — continuous, never binary;
- error is relative/normalized per term (|got-target| / max(1,|target|),
  capped at 1 per term), so exponential sequences don't crush linear ones
  and the metric is scale-invariant;
- parsimony: fitness loses `parsimony * tree_size`, so among equally
  correct formulas the shorter one always wins;
- exact hit = per-term error < EPS on every member; correctness is judged
  by values only, never by formula text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .evaluator import evaluate_all
from .nodes import Node, weighted_size

EPS = 1e-9
# Invalid candidates (division by zero, overflow) must not win selection;
# the exact value only has to sit below every scorable fitness.
INVALID_FITNESS = -1.0


def term_error(got: float, target: float) -> float:
    return min(1.0, abs(got - target) / max(1.0, abs(target)))


@dataclass(slots=True)
class Score:
    fitness: float
    error: float  # mean per-term error; math.inf for invalid candidates
    exact: bool
    values: list[float] | None
    errors: list[float] | None
    size: int


def score_candidate(node: Node, seq: list[float], parsimony: float) -> Score:
    tree_size = int(weighted_size(node))
    values = evaluate_all(node, len(seq))
    if values is None:
        return Score(INVALID_FITNESS, math.inf, False, None, None, tree_size)
    
    # Inlined term_error logic to avoid thousands of function calls per generation
    errors = [min(1.0, abs(got - target) / max(1.0, abs(target))) 
              for got, target in zip(values, seq)]
              
    error = sum(errors) / len(errors)
    exact = all(e < EPS for e in errors)
    fitness = (1.0 - error) - parsimony * tree_size
    return Score(fitness, error, exact, values, errors, tree_size)
