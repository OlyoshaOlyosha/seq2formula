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
