"""Tests for grammar v2: unary functions, mod, and the π/e constants."""

import math
import random

from engine import (BinOp, Const, GPConfig, GPEngine, Neg, Node, UnOp,
                    UNARY_FUNCS, from_dict, size, to_dict, to_text)
from engine.evaluator import evaluate
from engine.fitness import score_candidate
from engine.grammar import (BINARY_OPS, ensure_grammar, initial_population,
                            point_mutation, random_tree)
from engine.nodes import VAR, Var


def collect_ops(node):
    if isinstance(node, BinOp):
        yield node.op
        yield from collect_ops(node.left)
        yield from collect_ops(node.right)
    elif isinstance(node, UnOp):
        yield from collect_ops(node.child)


def collect_funcs(node):
    if isinstance(node, UnOp):
        yield node.name
        yield from collect_funcs(node.child)
    elif isinstance(node, BinOp):
        yield from collect_funcs(node.left)
        yield from collect_funcs(node.right)


# ---------- rendering ----------

def test_unop_text_rendering():
    assert to_text(UnOp("sqrt", VAR)) == "sqrt(n)"
    assert to_text(UnOp("fact", VAR)) == "n!"
    assert to_text(UnOp("fact", BinOp("+", VAR, VAR))) == "(n + n)!"
    assert to_text(UnOp("fact", UnOp("fact", VAR))) == "(n!)!"
    assert to_text(UnOp("fact", Neg(Const(3)))) == "(-3)!"
    assert to_text(BinOp("mod", VAR, Const(3))) == "n mod 3"
    assert to_text(Const(math.pi)) == "π"
    assert to_text(Const(math.e)) == "e"


def test_unop_json_round_trip():
    tree = UnOp("sqrt", BinOp("+", VAR, Const(math.pi)))
    assert to_text(from_dict(to_dict(tree))) == to_text(tree)


def test_from_dict_rejects_unknown_function():
    try:
        from_dict({"t": "f", "n": "evil", "x": {"t": "n"}})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown function accepted")


# ---------- evaluator ----------

def test_sqrt_domain_guard():
    assert evaluate(UnOp("sqrt", Const(-1)), 1) == math.inf
    assert evaluate(UnOp("sqrt", VAR), 4) == 2.0


def test_cbrt_is_real_for_negatives():
    assert evaluate(UnOp("cbrt", Const(-8)), 1) == -2.0
    assert evaluate(UnOp("cbrt", VAR), 27) == 3.0


def test_factorial_guards():
    assert evaluate(UnOp("fact", VAR), 5) == 120.0
    assert evaluate(UnOp("fact", VAR), 5.5) == math.inf   # non-integer
    assert evaluate(UnOp("fact", VAR), -3) == math.inf    # negative
    assert evaluate(UnOp("fact", VAR), 200) == math.inf   # out of range


def test_sin_cos_tan_ln():
    assert evaluate(UnOp("sin", VAR), 0) == 0.0
    assert abs(evaluate(UnOp("cos", VAR), 0) - 1) < 1e-12
    assert evaluate(UnOp("ln", VAR), math.e) == 1.0
    assert evaluate(UnOp("ln", Const(-1)), 1) == math.inf
    assert math.isfinite(evaluate(UnOp("tan", VAR), 1))


def test_mod_operation():
    assert evaluate(BinOp("mod", VAR, Const(3)), 7) == 1.0
    assert evaluate(BinOp("mod", VAR, VAR), 0) == math.inf  # mod 0


def test_wave_sequence_is_exact_by_values():
    # sin(pi * n / 2) at n = 1..8 -> 1 0 -1 0 1 0 -1 0 (fp noise << EPS)
    inner = BinOp("/", BinOp("*", Const(math.pi), VAR), Const(2))
    tree = UnOp("sin", inner)
    score = score_candidate(tree, [1, 0, -1, 0, 1, 0, -1, 0], 0.003)
    assert score.exact


def test_factorial_sequence_is_exact_by_values():
    score = score_candidate(UnOp("fact", VAR), [1, 2, 6, 24, 120, 720], 0.003)
    assert score.exact


# ---------- grammar ----------

def test_random_tree_respects_funcs_and_specials():
    rng = random.Random(11)
    for _ in range(200):
        tree = random_tree(rng, 5, "grow", ("+", "*"),
                           ("sqrt", "sin"), (math.pi,))
        assert all(f in ("sqrt", "sin") for f in collect_funcs(tree))
        assert _consts_within(tree, {math.pi} | set(range(-5, 6))
                              | {-1.5, -0.5, 0.5, 1.5})


def _consts_within(node, allowed):
    if isinstance(node, Const):
        return any(abs(node.value - v) < 1e-12 for v in allowed)
    if isinstance(node, BinOp):
        return _consts_within(node.left, allowed) and \
            _consts_within(node.right, allowed)
    if isinstance(node, UnOp):
        return _consts_within(node.child, allowed)
    return True  # Var


def test_no_funcs_means_v1_trees():
    rng = random.Random(12)
    for _ in range(100):
        tree = random_tree(rng, 5, "grow", BINARY_OPS)
        assert not any(True for _ in collect_funcs(tree))


def test_fact_init_only_on_useful_children():
    rng = random.Random(13)
    for _ in range(300):
        tree = random_tree(rng, 6, "grow", ("+", "*"), ("fact",))
        _assert_fact_children_simple(tree)


def _assert_fact_children_simple(node):
    if isinstance(node, UnOp):
        if node.name == "fact":
            child = node.child
            assert isinstance(child, Var) or (
                isinstance(child, Const) and 0 <= child.value <= 170
                and child.value == int(child.value))
        _assert_fact_children_simple(node.child)
    elif isinstance(node, BinOp):
        _assert_fact_children_simple(node.left)
        _assert_fact_children_simple(node.right)


def test_ensure_grammar_strips_disabled_function():
    rng = random.Random(14)
    tree = UnOp("sin", BinOp("+", VAR, VAR))
    fixed = ensure_grammar(tree, rng, ("+", "*"), ())  # sin disabled
    assert not any(True for _ in collect_funcs(fixed))
    assert to_text(fixed) == "n + n"


def test_point_mutation_swaps_functions():
    rng = random.Random(15)
    tree = UnOp("sin", VAR)
    seen = set()
    for _ in range(200):
        out = point_mutation(rng, tree, ("+",), ("cos", "tan"))
        # a root hit swaps the function; a child hit only alters the
        # argument and keeps the wrapper — both must stay valid grammar
        assert isinstance(out, UnOp)
        seen.add(out.name)
    assert seen <= {"sin", "cos", "tan"}
    assert seen & {"cos", "tan"}, "function swap never fired"


def test_depth_and_size_count_unop():
    tree = UnOp("sqrt", BinOp("+", VAR, VAR))
    assert size(tree) == 4
    from engine.nodes import depth
    assert depth(tree) == 3


# ---------- engine ----------

def test_default_config_searches_v1_space():
    eng = GPEngine([1, 2, 3, 4, 5])
    assert eng.funcs == ()
    assert eng.specials == ()
    for _ in range(3):
        eng.step()
    assert not any(any(True for _ in collect_funcs(t))
                   for t in eng.population)


def test_engine_offspring_respect_funcs():
    eng = GPEngine([1, 2, 3, 4, 5],
                   GPConfig(seed=21, allowed_funcs=("sqrt", "abs")))
    assert eng.funcs == ("sqrt", "abs")
    for _ in range(10):
        eng.step()
    for tree in eng.population:
        assert all(f in ("sqrt", "abs") for f in collect_funcs(tree))


def test_engine_offspring_respect_disabled_power_and_mod():
    eng = GPEngine([1, 2, 3, 4, 5],
                   GPConfig(seed=22, allowed_ops=("+", "-", "mod")))
    for _ in range(10):
        eng.step()
    for tree in eng.population:
        for op in collect_ops_deep(tree):
            assert op in ("+", "-", "mod")


def collect_ops_deep(node):
    if isinstance(node, BinOp):
        yield node.op
        yield from collect_ops_deep(node.left)
        yield from collect_ops_deep(node.right)
    elif isinstance(node, UnOp):
        yield from collect_ops_deep(node.child)


def test_full_v2_engine_solves_factorial_puzzle():
    eng = GPEngine([1, 2, 6, 24, 120, 720],
                   GPConfig(seed=23, population=400, max_generations=400,
                            allowed_funcs=UNARY_FUNCS))
    for _ in range(400):
        result = eng.step()
        if result.solved:
            break
    assert eng.solved, "n! should be reachable with fact enabled"


def test_all_functions_never_crash_engine():
    # Every draw path over every function must evaluate guarded: no
    # exception may escape step() whatever the population breeds.
    eng = GPEngine([3, 1, 4, 1, 5, 9],
                   GPConfig(seed=24, population=120, max_generations=15,
                            allowed_funcs=UNARY_FUNCS,
                            allowed_consts=("pi", "e"),
                            allowed_ops=BINARY_OPS))
    for _ in range(15):
        eng.step()
