"""Curve-mode server wiring: RunRequest validation, the CurveGP branch of
RunManager.start, and the data/curves.json gallery written by _finalize."""

import asyncio
import json

import pytest
from fastapi import HTTPException

import app as server
from app import RunManager, RunRequest
from engine import CurveGP

LINE = [[0.0, 0.0], [1.0, 1.0]]


def test_validate_points() -> None:
    with pytest.raises(HTTPException):
        RunManager._validate_points(None)
    with pytest.raises(HTTPException):
        RunManager._validate_points([[0.5, 0.5]])  # need >= 2
    with pytest.raises(HTTPException):
        RunManager._validate_points([[0.5, 0.5]] * 201)  # need <= 200
    with pytest.raises(HTTPException):
        RunManager._validate_points([[1.5, 0.1], [0.2, 0.2]])  # unit square
    with pytest.raises(HTTPException):
        RunManager._validate_points([[0.5], [0.2, 0.2]])  # each point [x, y]
    pts = RunManager._validate_points([[0.0, 0.0], [1.0, 1.0]])
    assert pts == [(0.0, 0.0), (1.0, 1.0)]


def test_curve_run_runs_and_lands_in_gallery(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(server, "CURVES_FILE", tmp_path / "curves.json")
    monkeypatch.setattr(
        server, "SETTINGS_FILE",
        _settings_file(tmp_path, {"population": 100, "max_generations": 10}))
    manager = RunManager()

    async def scenario() -> None:
        snap = manager.start(RunRequest(mode="curve", points=LINE, speed=240))
        assert snap["mode"] == "curve"
        assert isinstance(manager.engine, CurveGP)
        assert snap["points"] == LINE
        await asyncio.sleep(2.0)  # the loop should exhaust 10 generations
        assert manager.status == "finished"
        assert manager.finish_reason == "max_generations"
        await asyncio.sleep(0.1)

    asyncio.run(scenario())
    curves = json.loads((tmp_path / "curves.json").read_text(encoding="utf-8"))
    assert len(curves) == 1
    entry = curves[0]
    for key in ("ts", "points", "preset", "x_text", "x_ast", "y_text",
                "y_ast", "error", "generations", "time_s", "settings",
                "seed", "formula", "formula_ast", "solved"):
        assert key in entry
    assert entry["points"] == LINE
    assert entry["x_text"] and entry["y_text"]
    assert entry["solved"] is False


def test_multi_stroke_run_lands_in_gallery(tmp_path, monkeypatch) -> None:
    strokes = [[[0.0, 0.25], [1.0, 0.25]], [[0.0, 0.75], [1.0, 0.75]]]
    monkeypatch.setattr(server, "CURVES_FILE", tmp_path / "curves.json")
    monkeypatch.setattr(
        server, "SETTINGS_FILE",
        _settings_file(tmp_path, {"population": 100, "max_generations": 5}))
    manager = RunManager()

    async def scenario() -> None:
        snap = manager.start(RunRequest(mode="curve", strokes=strokes,
                                        speed=240, preset="Улыбка"))
        assert snap["mode"] == "curve"
        assert len(snap["strokes"]) == 2
        from engine import CurveGP
        assert isinstance(manager.engine, CurveGP)
        assert manager.engine.n_strokes == 2
        await asyncio.sleep(1.5)
        manager.stop(reason="test", notify=False)
        await asyncio.sleep(0.1)

    asyncio.run(scenario())
    curves = json.loads((tmp_path / "curves.json").read_text(encoding="utf-8"))
    entry = curves[0]
    assert entry["strokes"] == strokes
    assert entry["points"] == [[0.0, 0.25], [1.0, 0.25], [0.0, 0.75], [1.0, 0.75]]
    assert entry["preset"] == "Улыбка"
    assert len(entry["x_ast"]) == 2  # one pair per stroke


def test_stroke_validation() -> None:
    good = [[[0.0, 0.0], [1.0, 1.0]]]
    assert len(RunManager._validate_strokes(good)) == 1
    # a single-point stroke is valid (the engine pads it)
    assert len(RunManager._validate_strokes([[[0.5, 0.5]]])) == 1
    with pytest.raises(HTTPException):
        RunManager._validate_strokes(None)
    with pytest.raises(HTTPException):
        RunManager._validate_strokes([[[0.5, 0.5]]] * 7)  # stroke cap
    with pytest.raises(HTTPException):
        RunManager._validate_strokes([[[1.5, 0.1], [0.2, 0.2]]])  # unit square
    with pytest.raises(HTTPException):
        RunManager._validate_strokes([[[0.0], [1.0, 1.0]]])  # not [x, y] pairs
    with pytest.raises(HTTPException):
        RunManager._validate_strokes(
            [[[float(i) / 100, 0.5]] for i in range(201)])  # point cap


def test_sequence_mode_rejected_points_and_vice_versa(tmp_path) -> None:
    manager = RunManager()

    async def scenario() -> None:
        # curve request without points -> 422
        with pytest.raises(HTTPException):
            manager.start(RunRequest(mode="curve", speed=240))
        # sequence run still works and reports mode/points in the snapshot
        snap = manager.start(RunRequest(terms=[1, 2, 4, 8], speed=240))
        assert snap["mode"] == "sequence"
        assert snap["points"] is None
        assert isinstance(manager.engine, CurveGP) is False
        manager.stop(reason="test", notify=False)

    asyncio.run(scenario())


def _settings_file(tmp_path, overrides: dict):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(overrides), encoding="utf-8")
    return path
