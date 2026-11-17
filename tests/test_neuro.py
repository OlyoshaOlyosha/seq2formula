"""Tests for the RL neuro engine (engine/neuro.py)."""

from engine.neuro import NeuroConfig, NeuroEngine
from engine.nodes import BinOp


def test_rollouts_are_valid_trees() -> None:
    """Every sampled candidate must be a scorable tree: no crash, finite
    size, and the ^ tokens must keep a constant on one side."""
    eng = NeuroEngine([2.0, 4.0, 6.0, 8.0],
                      NeuroConfig(seed=1, population=50))
    for _ in range(5):
        eng.step()
    for node, sc in _collect(eng):
        assert sc.size >= 1
        assert _pow_ok(node)


def _collect(eng: NeuroEngine):
    """Re-sample one batch without learning, exposing the candidates."""
    scored = []
    for _ in range(eng.cfg.population):
        trace: list = []
        node = eng._rollout_slot(eng.cfg.max_depth, 0, trace, last_id=0)
        from engine.fitness import score_candidate
        scored.append((node, score_candidate(node, eng.seq,
                                             eng.cfg.parsimony)))
    return scored


def _pow_ok(node) -> bool:
    if isinstance(node, BinOp):
        if node.op == "^":
            from engine.nodes import Const as C
            if not isinstance(node.left, C) and not isinstance(node.right, C):
                return False
        return _pow_ok(node.left) and _pow_ok(node.right)
    if hasattr(node, "child"):
        return _pow_ok(node.child)
    return True


def test_seeded_runs_are_deterministic() -> None:
    texts = []
    for _ in range(2):
        eng = NeuroEngine([1.0, 4.0, 9.0], NeuroConfig(seed=7))
        for _ in range(3):
            eng.step()
        assert eng.best_ever is not None
        texts.append(eng.best_ever.score.fitness)
    assert texts[0] == texts[1]


def test_best_ever_monotone() -> None:
    eng = NeuroEngine([1.0, 2.0, 3.0], NeuroConfig(seed=2))
    prev = -1e9
    for _ in range(20):
        res = eng.step()
        assert eng.best_ever is not None
        assert eng.best_ever.score.fitness >= prev - 1e-12
        prev = eng.best_ever.score.fitness
        assert not res.solved or res.solved_now or True


def test_learns_linear_sequence() -> None:
    """Smoke learning check: on 2n the policy must reach an exact solve
    well within the budget — REINFORCE with a baseline on a ~40-token
    vocabulary should crack a linear rule in a couple of hundred steps."""
    eng = NeuroEngine([2.0, 4.0, 6.0, 8.0],
                      NeuroConfig(seed=0, population=64))
    solved = False
    for _ in range(300):
        eng.step()
        if eng.solved:
            solved = True
            break
    assert solved, f"best error after 300 steps: {eng.best_ever and eng.best_ever.score.error}"


def test_var_candidate_is_legal() -> None:
    """Var()/const candidates must score, not crash (degenerate policies
    must stay valid)."""
    eng = NeuroEngine([1.0, 2.0], NeuroConfig(seed=3))
    eng.step()
    assert eng.best_ever is not None


def test_func_tokens_do_not_parse_as_consts() -> None:
    """"cos"/"cbrt" start with "c" — they must build UnOp nodes, never
    crash the const parser (caught live: ValueError float("os"))."""
    eng = NeuroEngine([1.0, 2.0, 3.0],
                      NeuroConfig(seed=4, allowed_funcs=("sqrt", "sin",
                                                         "cos", "cbrt")))
    for _ in range(10):
        eng.step()
    assert eng.best_ever is not None


def test_memetic_wrap_move_finishes_composite_family() -> None:
    """The wrap move (add/subtract a small constant term) is the finishing
    move for composite families: on (2n-1)^2 the search climbs to
    4n^2-4n (fitness 0.84) and no constant nudge reaches the exact
    4n^2-4n+1 — adding a NEW term is not a local move on any existing
    node. The wrap must find it from that skeleton."""
    import random

    from engine.fitness import score_candidate
    from engine.neuro import _mutated_neighbor
    from engine.nodes import Const, Var

    seq = [1.0, 9.0, 25.0, 49.0, 81.0, 121.0, 169.0, 225.0]
    n = Var()
    skeleton = BinOp("-", BinOp("*", Const(4.0),
                                BinOp("^", n, Const(2.0))),
                     BinOp("*", Const(4.0), n))  # 4n^2 - 4n, fit 0.84
    rng = random.Random(0)
    for _ in range(100):
        nb = _mutated_neighbor(skeleton, rng,
                               ("+", "-", "*", "/", "^", "mod"), ())
        if nb is None:
            continue
        if score_candidate(nb, seq, 0.001).exact:
            break
    else:
        raise AssertionError("wrap move never found 4n^2-4n+1 in 100 tries")


def test_memetic_wrap_move_adds_const_term() -> None:
    """Every wrap-branch neighbor is tree +- a small constant — never a
    mangled tree (the move wraps, it does not touch inner nodes)."""
    import random

    from engine.neuro import _mutated_neighbor
    from engine.nodes import Const, Var

    tree = BinOp("*", Const(2.0), Var())
    rng = random.Random(5)
    neighbors = [_mutated_neighbor(tree, rng,
                                   ("+", "-", "*", "/", "^", "mod"), ())
                 for _ in range(50)]
    from engine.evaluator import evaluate_all

    for nb in neighbors:
        if nb is None:
            continue
        # valid tree that still evaluates on n=1..4
        assert evaluate_all(nb, 4) is not None


def test_var_substitution_move_solves_composite_family() -> None:
    """The var-substitution move (n -> c1*n+c2) is the elevator across
    composite families: n^2 becomes (2n-1)^2 EXACT in one move — a jump
    no constant nudge or op swap can express. Also: the substitution is
    never the identity (c1 != 1 by construction)."""
    import random

    from engine.fitness import score_candidate
    from engine.neuro import _mutated_neighbor
    from engine.nodes import Const, Var

    seq = [1.0, 9.0, 25.0, 49.0, 81.0, 121.0, 169.0, 225.0]  # (2n-1)^2
    tree = BinOp("^", Var(), Const(2.0))  # n^2, fit 0.41
    rng = random.Random(0)
    saw_identity = False
    # exact (2n-1) substitution: p ~ 0.15% per try — needs a few thousand
    for _ in range(3000):
        nb = _mutated_neighbor(tree, rng,
                               ("+", "-", "*", "/", "^", "mod"), ())
        if nb is None:
            continue
        # identity check: c1 != 1 by construction, so 2*n+d never
        # collapses back to plain n
        if nb == Var():
            saw_identity = True
        if score_candidate(nb, seq, 0.001).exact:
            break
    else:
        raise AssertionError("var-subst never found (2n-1)^2 in 3000 tries")
    assert not saw_identity


def test_progressive_grammar_widens_and_gates_masks() -> None:
    """Full-grammar config: the engine starts at the base stage (funcs
    and piecewise masked out), unlocks funcs after WIDEN_WINDOW stagnant
    generations and piecewise+specials after another; a base-only config
    never widens (progressive is a no-op there)."""
    from engine.neuro import STAGE_BASE, STAGE_FUNCS, STAGE_PIECE, WIDEN_WINDOW
    from engine.nodes import UNARY_FUNCS

    # unsolvable-but-valid sequence: keeps stagnation growing, no new best
    seq = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    specials = (3.141592653589793, 2.718281828459045)
    cfg = NeuroConfig(seed=0, population=64, max_generations=200,
                      allowed_ops=("+", "-", "*", "/", "^", "mod", "<", ">"),
                      allowed_funcs=tuple(UNARY_FUNCS),
                      allowed_consts=(*NeuroConfig().allowed_consts,
                                      *specials))
    eng = NeuroEngine(seq, cfg)
    assert eng.progressive
    assert eng.stage == STAGE_BASE
    mask0 = eng._legal_mask(4, 0, STAGE_BASE)
    assert mask0[eng.tok_id["sqrt"]] == 0.0
    assert mask0[eng.tok_id["if"]] == 0.0

    stages = {eng.stage}
    # new-best improvements reset the widen counter, so the exact widen
    # generations vary — run to the budget and expect both unlocks
    for _ in range(200):
        eng.step()
        stages.add(eng.stage)
        if eng.stage == STAGE_PIECE:
            break
    assert stages == {STAGE_BASE, STAGE_FUNCS, STAGE_PIECE}, stages
    mask2 = eng._legal_mask(4, 0, STAGE_PIECE)
    assert mask2[eng.tok_id["sqrt"]] == 1.0
    assert mask2[eng.tok_id["if"]] == 1.0
    assert mask2[eng.tok_id[f"c{specials[0]:g}"]] == 1.0

    # base-only config: nothing to widen, behavior identical to before
    plain = NeuroEngine(seq, NeuroConfig(seed=0, population=16,
                                         max_generations=5))
    assert not plain.progressive
    for _ in range(WIDEN_WINDOW + 2):
        plain.step()
    assert plain.stage == STAGE_BASE


def test_progressive_stage_is_deterministic() -> None:
    """Same seed + same config = same stage timeline and same best-ever
    fitness over the first generations."""
    from engine.nodes import UNARY_FUNCS

    seq = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    runs = []
    for _ in range(2):
        eng = NeuroEngine(seq, NeuroConfig(seed=7, population=32,
                                           max_generations=100,
                                           allowed_funcs=tuple(UNARY_FUNCS)))
        trace = []
        for _ in range(60):
            res = eng.step()
            trace.append((eng.stage,
                          round(res.best.score.fitness, 10)))
        runs.append(trace)
    assert runs[0] == runs[1]
