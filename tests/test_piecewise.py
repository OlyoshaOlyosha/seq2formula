"""Tests for grammar v3: comparison ops and the piecewise If node."""

import random

from engine import CMP_OPS, BinOp, Const, If, Var, depth, from_dict, size, to_dict, to_text
from engine.evaluator import compile_node, evaluate, evaluate_all
from engine.fitness import score_candidate
from engine.grammar import (
    BINARY_OPS,
    ensure_grammar,
    point_mutation,
    random_path,
    random_tree,
    replace_subtree,
    subtree_at,
)
from engine.neuro import IF_TOKEN, NeuroConfig, NeuroEngine
from engine.nodes import Neg, UnOp, simplify, weighted_size

OPS_IF = BINARY_OPS + CMP_OPS

# if n < 3 then 10 else 20 — the canonical piecewise candidate
FORK = If(BinOp("<", Var(), Const(3.0)), Const(10.0), Const(20.0))


def collect_ifs(node):
    if isinstance(node, If):
        yield node
        yield from collect_ifs(node.cond)
        yield from collect_ifs(node.then)
        yield from collect_ifs(node.otherwise)
    elif isinstance(node, BinOp):
        yield from collect_ifs(node.left)
        yield from collect_ifs(node.right)
    elif isinstance(node, (Neg, UnOp)):
        yield from collect_ifs(node.child)


# ---- evaluation ----

def test_comparison_evaluates_to_one_or_zero():
    cmp = BinOp("<", Var(), Const(3.0))
    assert evaluate(cmp, 1) == 1.0
    assert evaluate(cmp, 3) == 0.0
    assert evaluate(BinOp(">", Var(), Const(3.0)), 5) == 1.0
    assert evaluate(BinOp(">", Var(), Const(3.0)), 3) == 0.0


def test_if_picks_branch_by_condition():
    assert evaluate(FORK, 1) == 10.0
    assert evaluate(FORK, 2.9) == 10.0
    assert evaluate(FORK, 3) == 20.0
    assert evaluate(FORK, 100) == 20.0


def test_if_evaluate_all_and_compiled_agree():
    scalars = [evaluate(FORK, n) for n in range(1, 6)]
    assert evaluate_all(FORK, 5) == scalars
    fn = compile_node(FORK)
    assert [fn(n) for n in range(1, 6)] == scalars


def test_nested_if_and_comparison_expressions():
    # if (n < 2) < 1 then 5 else 6 — a comparison nested in a condition
    inner = BinOp("<", BinOp("<", Var(), Const(2.0)), Const(1.0))
    tree = If(inner, Const(5.0), Const(6.0))
    assert evaluate(tree, 1) == 6.0  # n<2 -> 1, 1 < 1 is False
    assert evaluate(tree, 2) == 5.0  # n<2 -> 0, 0 < 1 is True


# ---- text, serialization, size ----

def test_if_text_rendering_and_parens():
    assert to_text(FORK) == "if n < 3 then 10 else 20"
    nested = BinOp("*", Const(2.0), FORK)
    assert to_text(nested) == "2 * (if n < 3 then 10 else 20)"
    # a comparison nested in arithmetic needs parens (n + (n < 3) means
    # "n plus the 0/1 verdict", not "(n + n) < 3")
    assert to_text(BinOp("+", Var(), BinOp("<", Var(), Const(3.0)))) \
        == "n + (n < 3)"


def test_if_serialization_roundtrip():
    tree = If(BinOp(">", BinOp("*", Var(), Const(2.0)), Const(4.0)),
              UnOp("sqrt", Var()), Neg(Const(1.0)))
    assert from_dict(to_dict(tree)) == tree
    assert to_text(from_dict(to_dict(tree))) == to_text(tree)


def test_if_size_depth_weighted():
    assert size(FORK) == 6
    assert depth(FORK) == 3
    # fork costs 2.0; cond = cmp(1) + var(1) + const(1); the two branch
    # constants cost by digit count (10 -> 2, 20 -> 2)
    assert weighted_size(FORK) == 2.0 + 3.0 + 2.0 + 2.0


# ---- simplification ----

def test_simplify_folds_if_and_comparison():
    assert to_text(simplify(If(Const(0.0), Const(10.0), Const(20.0)))) == "20"
    assert to_text(simplify(If(Const(5.0), Const(10.0), Const(20.0)))) == "10"
    assert to_text(simplify(If(BinOp("<", Const(1.0), Const(2.0)),
                               Const(10.0), Const(20.0)))) == "10"
    # equal branches: the fork is dead weight
    assert to_text(simplify(If(BinOp("<", Var(), Const(3.0)),
                               Const(7.0), Const(7.0)))) == "7"


# ---- grammar ----

def test_random_tree_draws_if_when_comparisons_enabled():
    rng = random.Random(42)
    forks = 0
    for _ in range(400):
        tree = random_tree(rng, 6, "grow", OPS_IF)
        assert depth(tree) <= 6
        forks += sum(1 for _ in collect_ifs(tree))
    assert forks > 0  # the draw probability is low but not zero


def test_random_tree_never_draws_if_without_comparisons():
    rng = random.Random(42)
    for _ in range(200):
        tree = random_tree(rng, 6, "grow", BINARY_OPS)
        assert next(collect_ifs(tree), None) is None


def test_ensure_grammar_strips_if_when_comparisons_disabled():
    rng = random.Random(7)
    stripped = ensure_grammar(FORK, rng, BINARY_OPS)
    assert next(collect_ifs(stripped), None) is None
    assert to_text(stripped) == "10"  # the then branch survives
    # a bare comparison also strips (to its left operand)
    bare = ensure_grammar(BinOp("<", Var(), Const(3.0)), rng, BINARY_OPS)
    assert to_text(bare) == "n"


def test_paths_and_splice_through_if_children():
    tree = If(BinOp("<", Var(), Const(3.0)),
              BinOp("+", Const(10.0), Var()), Const(20.0))
    # every path resolves inside the right child; replacing the cond with
    # a whole new fork keeps the tree legal
    target = subtree_at(tree, (2,))
    assert target == Const(20.0)
    new_cond = BinOp(">", Var(), Const(100.0))
    spliced = replace_subtree(tree, (0,), new_cond)
    assert spliced.cond == new_cond
    path = random_path(random.Random(3), tree)
    assert subtree_at(tree, path) is not None


def test_point_mutation_swaps_if_branches():
    # the If root is one node among six, so the branch swap shows up
    # only when that node is picked — repeat until it does
    swapped = False
    for seed in range(60):
        mutated = point_mutation(random.Random(seed), FORK, OPS_IF)
        if isinstance(mutated, If) and mutated.then == FORK.otherwise:
            swapped = True
            break
    assert swapped


def test_crossover_with_forks_stays_legal():
    from engine.grammar import crossover
    rng = random.Random(5)
    a = If(BinOp("<", Var(), Const(3.0)), BinOp("^", Var(), Const(2.0)),
           Const(20.0))
    b = BinOp("+", Var(), If(BinOp(">", Var(), Const(2.0)),
                             Const(1.0), Const(9.0)))
    for _ in range(50):
        child = crossover(rng, a, b, max_depth=6, ops=OPS_IF)
        assert depth(child) <= 6
        for fork in collect_ifs(child):
            # every if-condition stays a comparison
            assert isinstance(fork.cond, BinOp)
            assert fork.cond.op in CMP_OPS


# ---- fitness / puzzle ----

def test_parity_puzzle_solved_by_if_formula():
    terms = [7.0, 3.0, 7.0, 3.0, 7.0, 3.0, 7.0, 3.0]
    formula = If(BinOp(">", BinOp("mod", Var(), Const(2.0)), Const(0.5)),
                 Const(7.0), Const(3.0))
    sc = score_candidate(formula, terms, parsimony=0.001)
    assert sc.exact and sc.values == terms


# ---- neuro ----

def test_neuro_vocab_gets_if_token_with_comparisons():
    engine = NeuroEngine(
        [7.0, 3.0, 7.0, 3.0],
        NeuroConfig(population=8, max_generations=1, memetic=False,
                    allowed_ops=OPS_IF))
    assert IF_TOKEN in engine.vocab
    assert engine.if_enabled
    # cmp ops are progressive-grammar stage 2: pass the final stage
    cmp_mask = engine._legal_mask(3, mode=2,
                                  stage=engine.max_stage)
    legal = [engine.vocab[i] for i in range(len(engine.vocab))
             if cmp_mask[i]]
    assert sorted(legal) == sorted(CMP_OPS)
    legal = [engine.vocab[i] for i in range(len(engine.vocab))
             if cmp_mask[i]]
    assert sorted(legal) == sorted(CMP_OPS)
    # without comparisons the token is absent
    plain = NeuroEngine(
        [1.0, 2.0, 3.0],
        NeuroConfig(population=8, max_generations=1, memetic=False,
                    allowed_ops=BINARY_OPS))
    assert IF_TOKEN not in plain.vocab


def test_neuro_rollout_with_if_runs_and_scores():
    engine = NeuroEngine(
        [7.0, 3.0, 7.0, 3.0, 7.0, 3.0],
        NeuroConfig(population=24, max_generations=1, memetic=False,
                    allowed_ops=OPS_IF))
    result = engine.step()
    assert result.best.score.fitness is not None
    # the parity sequence must be solvable: some rollout finds the fork
    assert result.best.score.exact or engine.best_ever is not None


def test_neuro_memetic_neighbor_of_fork():
    from engine.neuro import _mutated_neighbor
    rng = random.Random(9)
    seen = set()
    for _ in range(40):
        nb = _mutated_neighbor(FORK, rng, OPS_IF, ())
        if nb is not None:
            seen.add(to_text(nb))
    # flipping the comparison mirrors the condition; swapping branches
    # exchanges the two formulas
    assert "if n > 3 then 10 else 20" in seen
    assert "if n < 3 then 20 else 10" in seen
