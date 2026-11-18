"""Tests for tree generation and genetic operators."""

import random

from engine.grammar import crossover, ensure_grammar, initial_population, point_mutation, random_tree, subtree_mutation
from engine.nodes import VAR, BinOp, Const, Neg, Var, depth


def test_initial_population_respects_limits():
    rng = random.Random(1)
    population = initial_population(rng, 200, max_depth=6)
    assert len(population) == 200
    assert all(depth(t) <= 6 for t in population)
    assert all(isinstance(t, (BinOp, Const, Var, Neg)) for t in population)


def test_random_tree_terminals_only_at_zero_budget():
    rng = random.Random(2)
    for _ in range(20):
        tree = random_tree(rng, 0)
        assert isinstance(tree, (Const, Var))


def test_power_never_has_two_arbitrary_operands():
    rng = random.Random(3)
    for _ in range(200):
        tree = ensure_grammar(random_tree(rng, 6), rng)
        for node in walk(tree):
            if isinstance(node, BinOp) and node.op == "^":
                # legality: constant on either side (n^k / k^expr) or an
                # exponent of exactly n ((expr)^n, n^n). Banned: an
                # arbitrary base with an arbitrary non-n exponent.
                assert isinstance(node.left, Const) \
                    or isinstance(node.right, Const) \
                    or isinstance(node.right, Var)


def walk(node):
    yield node
    if isinstance(node, BinOp):
        yield from walk(node.left)
        yield from walk(node.right)
    elif isinstance(node, Neg):
        yield from walk(node.child)


def test_ensure_grammar_repairs_bare_power():
    rng = random.Random(4)
    # n ^ (n + 1) assembled by hand: an arbitrary exponent expression is
    # the one still-banned form; repair must give it a constant exponent
    broken = BinOp("^", VAR, BinOp("+", VAR, Const(1.0)))
    fixed = ensure_grammar(broken, rng)
    assert isinstance(fixed.right, Const)


def test_expression_base_power_solves_pell():
    from engine.fitness import score_candidate
    s2 = BinOp("^", Const(2.0), Const(0.5))
    pell = BinOp(
        "/", BinOp("-",
                   BinOp("^", BinOp("+", Const(1.0), s2), VAR),
                   BinOp("^", BinOp("-", Const(1.0), s2), VAR)),
        BinOp("*", Const(2.0), s2))
    sc = score_candidate(pell, [1.0, 2.0, 5.0, 12.0, 29.0, 70.0, 169.0], 40)
    assert sc.exact


def test_random_tree_draws_expression_base_powers():
    rng = random.Random(7)
    found = 0
    for _ in range(400):
        tree = random_tree(rng, 6, "grow", ("+", "-", "*", "/", "^", "mod"))
        for node in walk(tree):
            if isinstance(node, BinOp) and node.op == "^" \
                    and isinstance(node.right, Var):
                found += 1
    assert found > 0


def test_crossover_respects_depth_limit():
    rng = random.Random(5)
    a = initial_population(rng, 1, 6)[0]
    b = initial_population(rng, 1, 6)[0]
    for _ in range(50):
        child = crossover(rng, a, b, max_depth=4)
        assert depth(child) <= 4


def test_mutations_keep_depth_limit():
    rng = random.Random(6)
    tree = initial_population(rng, 1, 6)[0]
    for _ in range(50):
        assert depth(subtree_mutation(rng, tree, max_depth=5)) <= 5
        assert depth(point_mutation(rng, tree)) <= depth(tree)


def test_parents_not_mutated_by_operators():
    rng = random.Random(7)
    a = initial_population(rng, 1, 6)[0]
    snapshot = str(a)
    crossover(rng, a, initial_population(rng, 1, 6)[0])
    subtree_mutation(rng, a, 6)
    point_mutation(rng, a)
    assert str(a) == snapshot


def test_half_integer_constants_appear():
    rng = random.Random(8)
    trees = initial_population(rng, 400, 6)
    values = {n.value for t in trees for n in walk(t) if isinstance(n, Const)}
    assert any(v in (0.5, -0.5, 1.5, -1.5) for v in values)
    assert any(float(v).is_integer() for v in values)


def test_large_constants_appear():
    from engine.grammar import LARGE_CONSTS
    rng = random.Random(42)
    trees = initial_population(rng, 1000, 6)
    values = {n.value for t in trees for n in walk(t) if isinstance(n, Const)}
    assert any(abs(v) in set(LARGE_CONSTS) for v in values), \
        "expected at least one large constant in 1000 trees"
