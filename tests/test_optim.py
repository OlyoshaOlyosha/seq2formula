"""Equivalence tests for the performance rework (compiled evaluator,
k-th-node path selection, targeted splice repair)."""

import math
import random

from engine.evaluator import BIG, evaluate_all
from engine.grammar import (
    BINARY_OPS,
    crossover,
    initial_population,
    random_path,
    random_tree,
    shrink_mutation,
    subtree_at,
    subtree_mutation,
)
from engine.nodes import UNARY_FUNCS, VAR, BinOp, Const, Neg, Node, UnOp, Var, size

# -- reference interpreter: the straightforward recursive evaluator ---------

def _ref_eval(node: Node, n: float) -> float:
    if isinstance(node, Const):
        return node.value
    if isinstance(node, Var):
        return n
    if isinstance(node, Neg):
        return -_ref_eval(node.child, n)
    if isinstance(node, UnOp):
        import engine.evaluator as ev
        return ev.UNARY_EVAL[node.name](_ref_eval(node.child, n))
    left = _ref_eval(node.left, n)
    right = _ref_eval(node.right, n)
    if node.op == "+":
        return left + right
    if node.op == "-":
        return left - right
    if node.op == "*":
        return left * right
    if node.op == "/":
        return left / right
    if node.op == "mod":
        return left % right
    return left ** right


def _ref_all(node: Node, length: int):
    values = []
    for i in range(1, length + 1):
        try:
            v = _ref_eval(node, i)
        except (ZeroDivisionError, OverflowError, ValueError, TypeError):
            return None
        if isinstance(v, complex) or not math.isfinite(v) or abs(v) > BIG:
            return None
        values.append(v)
    return values


def test_compiled_matches_reference():
    rng = random.Random(11)
    for funcs in ((), UNARY_FUNCS):
        for _ in range(300):
            tree = random_tree(rng, 6, "grow", BINARY_OPS, funcs, (math.pi,))
            assert evaluate_all(tree, 9) == _ref_all(tree, 9)


def test_compiled_matches_reference_on_population():
    # whole-run check: every scored candidate's values must equal the
    # reference values (or both invalid)
    from engine import GPConfig, GPEngine
    rng = random.Random(12)
    eng = GPEngine([2, 4, 5, 4, 1], GPConfig(seed=5))
    for _ in range(15):
        for node in eng.population:
            got = evaluate_all(node, 5)
            assert got == _ref_all(node, 5)
        eng.step()


# -- random_path: uniform preorder selection --------------------------------

def collect_paths(node: Node, path: tuple = ()):
    yield path
    if isinstance(node, BinOp):
        yield from collect_paths(node.left, path + (0,))
        yield from collect_paths(node.right, path + (1,))
    elif isinstance(node, (Neg, UnOp)):
        yield from collect_paths(node.child, path + (0,))


def test_random_path_hits_every_node_and_is_valid():
    rng = random.Random(21)
    for _ in range(50):
        tree = random_tree(rng, 6, "grow", BINARY_OPS, UNARY_FUNCS)
        paths = set(collect_paths(tree))
        assert len(paths) == size(tree)
        for _ in range(60):
            path = random_path(rng, tree)
            assert path in paths            # a real node of this tree
            subtree_at(tree, path)          # resolvable without overshoot


def test_random_path_is_uniform():
    rng = random.Random(22)
    tree = BinOp("+", BinOp("+", VAR, VAR), VAR)  # 5 nodes
    counts = {p: 0 for p in collect_paths(tree)}
    for _ in range(5000):
        counts[random_path(rng, tree)] += 1
    expected = 5000 / len(counts)
    assert all(c > expected * 0.7 for c in counts.values())


# -- targeted repair: operators never emit invalid grammar -------------------

def test_ops_never_break_power_rule():
    rng = random.Random(31)
    for _ in range(300):
        tree = random_tree(rng, 6, "grow")
        assert _power_ok(tree), "random_tree produced an invalid ^"
        for _ in range(20):
            m = subtree_mutation(rng, tree, 6)
            assert _power_ok(m), "subtree_mutation broke the ^ rule"
            s = shrink_mutation(rng, tree)
            assert _power_ok(s), "shrink_mutation broke the ^ rule"


def test_crossover_offspring_respect_power_rule():
    rng = random.Random(32)
    pop = initial_population(rng, 40, 6)
    for _ in range(300):
        a, b = rng.choice(pop), rng.choice(pop)
        child = crossover(rng, a, b, 6)
        assert _power_ok(child), "crossover broke the ^ rule"


def _power_ok(node: Node) -> bool:
    if isinstance(node, BinOp):
        if node.op == "^" and not isinstance(node.left, Const) \
                and not isinstance(node.right, Const):
            return False
        return _power_ok(node.left) and _power_ok(node.right)
    if isinstance(node, (Neg, UnOp)):
        return _power_ok(node.child)
    return True


# -- end-to-end sanity: searches still converge ------------------------------

def test_engines_still_solve_basics():
    from engine import GPConfig, GPEngine
    eng = GPEngine([2, 4, 6, 8, 10], GPConfig(seed=7))
    for _ in range(300):
        if eng.step().solved:
            break
    assert eng.solved

    eng = GPEngine([1, 2, 4, 8, 16, 32], GPConfig(seed=8))
    for _ in range(1200):
        result = eng.step()
        if result.solved and eng.generation - eng.solved_gen >= 120:
            break  # shrink phase over: shortening had its chance
    assert eng.solved
    assert eng.shortest_exact is not None
    assert size(eng.shortest_exact.node) <= 5  # shrink keeps formulas short
