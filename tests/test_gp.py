"""Tests for the GP loop: result shape, convergence, flags."""

import time

from engine import GPConfig, GPEngine
from engine.gp import SHRINK_GENERATIONS


def make_engine(seq, **overrides):
    return GPEngine([float(x) for x in seq], GPConfig(seed=11, **overrides))


def test_step_returns_generation_payload():
    eng = make_engine([1, 2, 3, 4])
    result = eng.step()
    payload = result.payload()
    assert payload["type"] == "gen"
    assert payload["gen"] == 0
    assert payload["best"]["values"] == payload["best"]["values"]
    assert set(payload["stats"]) >= {
        "best_fitness", "mean_fitness", "mean_error", "mean_size",
        "size_hist", "diversity", "valid_frac"}
    assert len(payload["stats"]["size_hist"]) == 20


def test_population_size_stable():
    eng = make_engine([1, 2, 3, 4])
    for _ in range(5):
        eng.step()
    assert len(eng.population) == eng.cfg.population


def test_linear_sequence_converges_fast():
    eng = make_engine([1, 2, 3, 4, 5, 6])
    for _ in range(60):
        result = eng.step()
        if result.solved:
            break
    assert eng.solved, "n must converge within 60 generations"
    assert eng.shortest_exact is not None
    assert eng.shortest_exact.score.exact


def test_done_after_shrink_phase():
    eng = make_engine([1, 2, 3, 4, 5])
    while not eng.done:
        eng.step()
    assert eng.solved
    assert eng.generation <= eng.solved_gen + SHRINK_GENERATIONS + 1


def test_done_at_max_generations():
    eng = make_engine([1, 4, 9, 16, 25, 36, 49], max_generations=3)
    steps = 0
    while not eng.done:
        eng.step()
        steps += 1
    assert steps == 3


def test_generation_step_is_fast_enough():
    # A generation must stay well under 100 ms so the dashboard can
    # stream at a watchable pace without starving the event loop.
    eng = make_engine([2, 4, 8, 16, 32], population=300)
    eng.step()  # warm-up
    started = time.perf_counter()
    eng.step()
    assert (time.perf_counter() - started) < 0.1


def test_curiosities_only_almost_correct():
    eng = make_engine([1, 2, 4, 8, 16], max_generations=30)
    while not eng.done:
        eng.step()
    for entry in eng.curiosity_payloads():
        formula = entry["formula"]
        assert not formula["exact"]
        assert formula["error"] < 0.2
        assert formula["size"] >= 8
