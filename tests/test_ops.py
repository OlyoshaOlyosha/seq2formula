"""Tests for the operator-restriction feature (allowed_ops)."""

import random

from engine import GPConfig, GPEngine
from engine.grammar import (BINARY_OPS, ensure_grammar, initial_population,
                            point_mutation, random_tree, subtree_mutation)
from engine.nodes import BinOp, VAR, depth


def collect_ops(node):
    if isinstance(node, BinOp):
        yield node.op
        yield from collect_ops(node.left)
        yield from collect_ops(node.right)


def test_random_tree_only_uses_allowed_ops():
    rng = random.Random(1)
    for _ in range(200):
        tree = random_tree(rng, 5, "grow", ("+", "*"))
        assert all(op in ("+", "*") for op in collect_ops(tree))


def test_initial_population_respects_ops():
    rng = random.Random(2)
    population = initial_population(rng, 100, 6, ("+", "-"))
    assert all(all(op in ("+", "-") for op in collect_ops(t))
               for t in population)


def test_engine_population_and_offspring_respect_ops():
    default = GPEngine([1, 2, 3, 4, 5])
    assert default.ops == BINARY_OPS  # default: everything

    restricted = GPEngine([1, 2, 3, 4, 5],
                          GPConfig(seed=3, allowed_ops=("+", "*")))
    for _ in range(10):
        restricted.step()
    assert all(all(op in ("+", "*") for op in collect_ops(t))
               for t in restricted.population)


def test_ensure_grammar_swaps_bare_power_when_disabled():
    rng = random.Random(4)
    broken = BinOp("^", VAR, VAR)  # n^n: illegal under any rule
    fixed = ensure_grammar(broken, rng, ("+", "-"))
    assert fixed.op in ("+", "-")


def test_mutations_respect_ops():
    rng = random.Random(5)
    tree = initial_population(rng, 1, 6, ("+", "-"))[0]
    for _ in range(50):
        mutated = subtree_mutation(rng, tree, 5, ("+", "-"))
        assert depth(mutated) <= 5
        assert all(op in ("+", "-") for op in collect_ops(mutated))
        point = point_mutation(rng, tree, ("+", "-"))
        assert all(op in ("+", "-") for op in collect_ops(point))


def test_addition_only_engine_still_solves_linear():
    eng = GPEngine([2, 4, 6, 8, 10], GPConfig(seed=6, allowed_ops=("+",)))
    for _ in range(200):
        result = eng.step()
        if result.solved:
            break
    assert eng.solved  # n + n needs only +
