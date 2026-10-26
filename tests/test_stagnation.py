"""Tests for the anti-stagnation shock mechanism."""


from engine import GPConfig, GPEngine
from engine.nodes import UnOp


def collect_func_names(node):
    if isinstance(node, UnOp):
        yield node.name
        yield from collect_func_names(node.child)
    else:
        for child in getattr(node, "child", None), \
                getattr(node, "left", None), getattr(node, "right", None):
            if child is not None:
                yield from collect_func_names(child)


def test_shock_fires_periodically_when_stuck():
    # 2^n is unsolvable without ^: the run grinds at an error floor and
    # must keep shocking instead of circling the same formulas forever
    eng = GPEngine([2, 4, 8, 16, 32, 64],
                   GPConfig(seed=9, stagnation_window=5, shock_fraction=0.5,
                            allowed_ops=("+", "-", "*", "/")))
    shocks = 0
    was_shocking = False
    for _ in range(60):
        result = eng.step()
        # a shock now lasts multiple generations: count the START of an
        # episode, not every generation inside it
        if result.shock and not was_shocking:
            shocks += 1
        was_shocking = result.shock
    assert eng.shocks == shocks
    assert shocks >= 2, f"expected repeated shocks, got {shocks}"


def test_shock_can_be_disabled():
    eng = GPEngine([2, 4, 8, 16, 32, 64],
                   GPConfig(seed=9, stagnation_window=0,
                            allowed_ops=("+", "-", "*", "/")))
    for _ in range(40):
        eng.step()
    assert eng.shocks == 0


def test_shock_offspring_respect_grammar():
    eng = GPEngine([2, 4, 8, 16, 32, 64],
                   GPConfig(seed=10, stagnation_window=5, shock_fraction=1.0,
                            allowed_funcs=("fact",),
                            allowed_consts=("pi",)))
    for _ in range(20):
        eng.step()
    for node in eng.population:
        assert all(name == "fact" for name in collect_func_names(node))


def test_shock_does_not_fire_after_solving():
    eng = GPEngine([2, 4, 6, 8, 10], GPConfig(seed=11, stagnation_window=3))
    for _ in range(300):
        result = eng.step()
        if result.solved and eng.generation - eng.solved_gen >= 2:
            break
    assert eng.solved
    # after the solution, shocks are off (shrink phase breeds from the
    # exact pool; a shock would throw the found formula away)
    for _ in range(50):
        assert not eng.step().shock
