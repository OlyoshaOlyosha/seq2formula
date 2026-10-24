"""Tests for the fitness layer: closeness, parsimony, exact hit."""

import math

from engine.fitness import (EPS, INVALID_FITNESS, score_candidate,
                            term_error)
from engine.nodes import BinOp, Const, Neg, Var, VAR


def B(op, left, right):
    return BinOp(op, left, right)


def C(v):
    return Const(float(v))


SEQ = [2.0, 4.0, 8.0, 16.0, 32.0]  # 2^n


def test_exact_hit_by_values():
    score = score_candidate(B("^", C(2), VAR), SEQ, parsimony=0.001)
    assert score.exact
    assert score.error == 0.0
    assert score.values == SEQ


def test_equivalent_formulas_both_exact():
    # n + n and 2*n are the same answer by values, never by text
    seq = [2.0, 4.0, 6.0, 8.0]
    for tree in (B("+", VAR, VAR), B("*", C(2), VAR), B("*", VAR, C(2))):
        assert score_candidate(tree, seq, parsimony=0.001).exact


def test_error_is_relative_not_raw():
    # Same absolute miss, huge target: relative error must be small
    assert term_error(1001.0, 1000.0) < 0.002
    # ...and tiny target: normalized by max(1, |target|)
    assert term_error(0.5, 0.0) == 0.5


def test_term_error_capped_at_one():
    assert term_error(100.0, 1.0) == 1.0


def test_error_scale_invariance():
    # The metric is invariant under scaling both sides by the same c
    assert term_error(3.0, 2.0) == term_error(3000.0, 2000.0)
    assert term_error(1.5, 2.0) == term_error(1500.0, 2000.0)


def test_closer_scores_higher():
    seq = [1.0, 2.0, 3.0, 4.0]
    near = score_candidate(B("+", VAR, C(1)), seq, parsimony=0.0)   # off by 1
    exact = score_candidate(VAR, seq, parsimony=0.0)
    assert exact.fitness > near.fitness


def test_parsimony_prefers_shorter_at_equal_error():
    seq = [2.0, 4.0, 6.0, 8.0]
    short = score_candidate(B("*", C(2), VAR), seq, parsimony=0.01)
    # (n + n) * 1: same values, more nodes
    long = score_candidate(B("*", B("+", VAR, VAR), C(1)), seq, parsimony=0.01)
    assert short.error == long.error == 0.0
    assert short.fitness > long.fitness


def test_invalid_candidate_gets_worst_fitness():
    score = score_candidate(B("/", C(1), B("-", VAR, VAR)), SEQ, parsimony=0.001)
    assert score.values is None
    assert score.error == math.inf
    assert score.fitness == INVALID_FITNESS
