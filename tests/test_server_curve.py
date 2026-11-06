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
