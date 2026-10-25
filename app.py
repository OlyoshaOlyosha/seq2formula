"""FastAPI server for seq2formula.

REST controls the search run (start / pause / step / speed / stop) and
serves the puzzle library, history, hall of fame and curiosities; the
WebSocket streams live generation events to the dashboard. All state is
in-process (single user, single active run); solved runs are appended
to data/history.json, curiosities to data/curiosities.json.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import GPConfig, GPEngine, UNARY_FUNCS, from_dict, to_dict, to_text
from engine.grammar import BINARY_OPS
from engine.gp import SHRINK_GENERATIONS
from engine.nodes import special_value

# Grammar feature tokens the dashboard may send in RunRequest.ops: binary
# operators, unary functions and special constants, one flat list — the
# grouping into picker sections is purely a UI concern.
CONST_TOKENS = ("pi", "e")
ALL_TOKENS = frozenset(BINARY_OPS) | frozenset(UNARY_FUNCS) | frozenset(CONST_TOKENS)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HISTORY_FILE = DATA_DIR / "history.json"
CURIOSITY_FILE = DATA_DIR / "curiosities.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
HISTORY_CAP = 500        # solved runs kept in the file
CURIOSITY_CAP = 60       # gallery entries kept across all runs
CURIOSITY_PER_RUN = 4    # best entries saved from one run

MIN_TERMS, MAX_TERMS = 2, 32
TERM_ABS_LIMIT = 1e9
DEFAULT_GPS = 10.0

# Engine knobs the Настройки tab can change; a fresh store starts here.
# Population/parsimony/max_generations default a RUN when the request does
# not carry them; the rest configure GPConfig directly.
DEFAULT_SETTINGS = {
    "population": 500,
    "max_generations": 10000,
    "speed_max": 60,          # upper bound of the dashboard speed slider
    "parsimony": 0.003,
    "max_depth": 6,
    "tournament": 3,
    "p_crossover": 0.85,
    "p_mutate": 0.25,
    "elitism": 2,
    "shrink_generations": 120,
    "shrink_parsimony": 0.02,
}


def load_settings() -> dict:
    stored = _load_json(SETTINGS_FILE, {})
    settings = dict(DEFAULT_SETTINGS)
    settings.update({k: v for k, v in stored.items() if k in DEFAULT_SETTINGS})
    return settings

app = FastAPI(title="seq2formula")


# --------------------------------------------------------------------------
# Small JSON-file storage (single user, single writer: plain save is fine)
# --------------------------------------------------------------------------

def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _seq_key(terms: list[float]) -> str:
    return ",".join(f"{t:.6g}" for t in terms)


def load_puzzles() -> list[dict]:
    data = _load_json(ROOT / "sequences.json", {"puzzles": []})
    return data.get("puzzles", [])


def solved_sequences() -> set[str]:
    return {entry["seq_key"] for entry in _load_json(HISTORY_FILE, [])
            if entry.get("solved")}


def hall_of_fame() -> list[dict]:
    """Shortest formula ever found per sequence, derived from history."""
    best: dict[str, dict] = {}
    for entry in _load_json(HISTORY_FILE, []):
        if not entry.get("solved"):
            continue
        key = entry["seq_key"]
        if key not in best or entry["size"] < best[key]["size"]:
            best[key] = entry
    return sorted(best.values(), key=lambda e: e["ts"], reverse=True)


# --------------------------------------------------------------------------
# Run manager: one active search, paced by asyncio, events to all sockets
# --------------------------------------------------------------------------

class RunRequest(BaseModel):
    terms: list[float] | None = None
    puzzle_id: str | None = None
    population: int | None = Field(default=None, ge=20, le=2000)
    parsimony: float | None = Field(default=None, ge=0.0, le=0.1)
    max_generations: int | None = Field(default=None, ge=10, le=100000)
    speed: float = Field(default=DEFAULT_GPS, ge=0.5, le=240)
    ops: list[str] | None = None  # feature tokens (ops/funcs/consts); None = all


class SettingsRequest(BaseModel):
    population: int = Field(ge=20, le=2000)
    max_generations: int = Field(ge=10, le=100000)
    speed_max: float = Field(ge=10, le=240)
    parsimony: float = Field(ge=0.0, le=0.1)
    max_depth: int = Field(ge=3, le=12)
    tournament: int = Field(ge=2, le=16)
    p_crossover: float = Field(ge=0.0, le=1.0)
    p_mutate: float = Field(ge=0.0, le=1.0)
    elitism: int = Field(ge=0, le=20)
    shrink_generations: int = Field(ge=0, le=2000)
    shrink_parsimony: float = Field(ge=0.0, le=0.2)


class ControlRequest(BaseModel):
    action: str = Field(pattern="^(pause|resume|step|stop)$")


class SpeedRequest(BaseModel):
    gps: float = Field(ge=0.5, le=240)


class RunManager:
    def __init__(self) -> None:
        self.engine: GPEngine | None = None
        self.task: asyncio.Task | None = None
        self.status: str = "idle"  # idle | running | paused | finished
        self.delay: float = 1.0 / DEFAULT_GPS
        self.step_once = False
        self.started_at: float = 0.0
        self.solved_wall_time: float | None = None
        self.puzzle: dict | None = None
        self.terms: list[float] = []
        self.last_gen: dict | None = None
        self.finish_reason: str | None = None
        self.clients: set[WebSocket] = set()

    # -- websocket plumbing ------------------------------------------------

    async def broadcast(self, payload: dict) -> None:
        message = json.dumps(payload, ensure_ascii=False)
        dead: list[WebSocket] = []
        for ws in self.clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    # -- run lifecycle -------------------------------------------------------

    def start(self, request: RunRequest) -> dict:
        terms, puzzle = self._resolve_terms(request)
        self.stop(reason="replaced", notify=False)
        allowed_ops, allowed_funcs, allowed_consts = \
            self._validate_grammar(request.ops)
        settings = load_settings()
        config = GPConfig(
            population=(request.population if request.population is not None
                        else settings["population"]),
            parsimony=(request.parsimony if request.parsimony is not None
                       else settings["parsimony"]),
            max_generations=(request.max_generations
                             if request.max_generations is not None
                             else settings["max_generations"]),
            max_depth=settings["max_depth"],
            tournament=settings["tournament"],
            p_crossover=settings["p_crossover"],
            p_mutate=settings["p_mutate"],
            elitism=settings["elitism"],
            shrink_generations=settings["shrink_generations"],
            shrink_parsimony=settings["shrink_parsimony"],
            allowed_ops=allowed_ops,
            allowed_funcs=allowed_funcs,
            allowed_consts=allowed_consts,
        )
        self.engine = GPEngine(terms, config)
        self.status = "running"
        self.delay = 1.0 / request.speed
        self.step_once = False
        self.started_at = time.perf_counter()
        self.solved_wall_time = None
        self.puzzle = puzzle
        self.terms = terms
        self.last_gen = None
        self.finish_reason = None
        self.task = asyncio.get_running_loop().create_task(self._loop())
        return self.snapshot()

    @staticmethod
    def _validate_grammar(
            ops: list[str] | None,
    ) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None,
               tuple[str, ...] | None]:
        """Split the flat feature-token list into (binary ops, unary
        funcs, special consts); None members mean "the full set"."""
        if ops is None:
            return None, None, None
        if not ops:
            raise HTTPException(422, "at least one feature must be enabled")
        bad = [token for token in ops if token not in ALL_TOKENS]
        if bad:
            raise HTTPException(422, f"unknown grammar tokens: {bad}")
        ordered = tuple(dict.fromkeys(ops))  # dedupe, keep order
        bin_ops = tuple(t for t in ordered if t in BINARY_OPS) or None
        funcs = tuple(t for t in ordered if t in UNARY_FUNCS) or ()
        consts = tuple(t for t in ordered if t in CONST_TOKENS) or ()
        if bin_ops is None:
            raise HTTPException(422, "at least one operator must be enabled")
        return bin_ops, funcs, consts

    def _resolve_terms(self, request: RunRequest) -> tuple[list[float], dict | None]:
        if request.puzzle_id is not None:
            for puzzle in load_puzzles():
                if puzzle["id"] == request.puzzle_id:
                    return [float(t) for t in puzzle["terms"]], puzzle
            raise HTTPException(404, f"unknown puzzle: {request.puzzle_id}")
        if not request.terms:
            raise HTTPException(422, "provide terms or puzzle_id")
        if not (MIN_TERMS <= len(request.terms) <= MAX_TERMS):
            raise HTTPException(422, f"sequence length must be {MIN_TERMS}..{MAX_TERMS}")
        for term in request.terms:
            if not math.isfinite(term) or abs(term) > TERM_ABS_LIMIT:
                raise HTTPException(422, "terms must be finite numbers within 1e9")
        return [float(t) for t in request.terms], None

    def control(self, action: str) -> dict:
        if self.engine is None or self.status == "idle":
            raise HTTPException(409, "no active run")
        if action == "pause" and self.status == "running":
            self.status = "paused"
        elif action == "resume" and self.status in ("paused", "running"):
            self.status = "running"
        elif action == "step":
            if self.status == "finished":
                raise HTTPException(409, "run already finished")
            self.status = "paused"
            self.step_once = True
        elif action == "stop":
            self.stop(reason="stopped_by_user")
            return self.snapshot()
        return self.snapshot()

    def stop(self, reason: str, notify: bool = True) -> None:
        if self.task is not None:
            self.task.cancel()
            self.task = None
        if self.engine is not None and self.status not in ("idle", "finished"):
            self.status = "finished"
            self.finish_reason = reason
            if notify:
                asyncio.get_running_loop().create_task(self._finalize())
        elif self.engine is None:
            self.status = "idle"

    def snapshot(self) -> dict:
        return {
            "type": "status",
            "status": self.status,
            "sequence": self.terms,
            "puzzle": ({k: self.puzzle[k] for k in ("id", "tier", "name")}
                       if self.puzzle else None),
            "generation": self.engine.generation if self.engine else 0,
            "population": self.engine.cfg.population if self.engine else 0,
            "delay": self.delay,
            "best": self.last_gen["best"] if self.last_gen else None,
            "solved": bool(self.engine and self.engine.solved),
            "finish_reason": self.finish_reason,
        }

    # -- the paced loop ------------------------------------------------------

    async def _loop(self) -> None:
        try:
            while True:
                wants_step = (self.status == "running"
                              or (self.step_once and self.status == "paused"))
                self.step_once = False
                if wants_step and self.engine is not None:
                    loop = asyncio.get_event_loop()
                    started = time.perf_counter()
                    result = await loop.run_in_executor(None, self.engine.step)
                    self.last_gen = result.payload()
                    await self.broadcast(self.last_gen)
                    if result.solved_now:
                        self.solved_wall_time = time.perf_counter() - self.started_at
                    if self.engine.done:
                        await self._finalize()
                        return
                    await asyncio.sleep(max(0.0, self.delay
                                            - (time.perf_counter() - started)))
                else:
                    await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass

    async def _finalize(self) -> None:
        """Persist the finished run and announce the outcome."""
        assert self.engine is not None
        solved = self.engine.solved
        entry = None
        if solved:
            best = self.engine.shortest_exact or self.engine.best_ever
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seq_key": _seq_key(self.terms),
                "sequence": self.terms,
                "puzzle": self.puzzle["name"] if self.puzzle else None,
                "formula": to_text(best.node),
                "formula_ast": to_dict(best.node),
                "size": best.score.size,
                "values": ([round(v, 6) for v in best.score.values]
                           if best.score.values else None),
                "generations": self.engine.solved_gen + 1,
                "time_s": round(self.solved_wall_time or
                                (time.perf_counter() - self.started_at), 3),
                "solved": True,
            }
            history = _load_json(HISTORY_FILE, [])
            history.append(entry)
            _save_json(HISTORY_FILE, history[-HISTORY_CAP:])
            self._save_curiosities()
        self.status = "finished"
        self.finish_reason = ("solved" if solved else
                              self.finish_reason or "max_generations")
        await self.broadcast({
            "type": "solved" if solved else "finished",
            "reason": self.finish_reason,
            "formula": entry["formula_ast"] if entry else None,
            "formula_text": entry["formula"] if entry else None,
            "size": entry["size"] if entry else None,
            "generations": entry["generations"] if entry else self.engine.generation,
            "time_s": entry["time_s"] if entry else None,
            "hall_entry": entry,
            "shortest_text": (to_text(self.engine.shortest_exact.node)
                              if self.engine.shortest_exact else None),
        })

    def _save_curiosities(self) -> None:
        """Merge this run's best almost-correct monsters into the gallery."""
        fresh = self.engine.curiosity_payloads(limit=CURIOSITY_PER_RUN)
        if not fresh:
            return
        gallery = _load_json(CURIOSITY_FILE, [])
        known = {item["formula_text"] for item in gallery}
        for item in fresh:
            text = item["formula"]["text"]
            if text in known:
                continue
            known.add(text)
            gallery.append({
                "formula_text": text,
                "formula_ast": item["formula"]["ast"],
                "size": item["formula"]["size"],
                "error": item["formula"]["error"],
                "values": item["formula"]["values"],
                "sequence": self.terms,
                "gen": item["gen"],
            })
        gallery.sort(key=lambda item: item["size"], reverse=True)
        _save_json(CURIOSITY_FILE, gallery[:CURIOSITY_CAP])


manager = RunManager()


# --------------------------------------------------------------------------
# REST API
# --------------------------------------------------------------------------

@app.get("/api/puzzles")
def get_puzzles() -> dict:
    solved = solved_sequences()
    puzzles = []
    for puzzle in load_puzzles():
        item = dict(puzzle)
        if _seq_key([float(t) for t in puzzle["terms"]]) not in solved:
            item["reference"] = None  # revealed only once solved
        puzzles.append(item)
    return {"puzzles": puzzles}


@app.get("/api/history")
def get_history() -> dict:
    return {"history": list(reversed(_load_json(HISTORY_FILE, [])))[:100]}


@app.get("/api/hall")
def get_hall() -> dict:
    return {"hall": hall_of_fame()}


@app.get("/api/curiosities")
def get_curiosities() -> dict:
    return {"curiosities": _load_json(CURIOSITY_FILE, [])}


@app.get("/api/settings")
def get_settings() -> dict:
    return {"settings": load_settings(), "defaults": DEFAULT_SETTINGS}


@app.post("/api/settings")
def post_settings(request: SettingsRequest) -> dict:
    _save_json(SETTINGS_FILE, request.model_dump())
    return {"settings": load_settings(), "defaults": DEFAULT_SETTINGS}


@app.get("/api/status")
def get_status() -> dict:
    return manager.snapshot()


@app.post("/api/run")
async def post_run(request: RunRequest) -> dict:
    snapshot = manager.start(request)
    # Tell every connected page a new run started (multi-tab sync)
    asyncio.get_running_loop().create_task(manager.broadcast(snapshot))
    return snapshot


@app.post("/api/control")
async def post_control(request: ControlRequest) -> dict:
    snapshot = manager.control(request.action)
    # Pages must learn pause/resume/step/stop even when another tab
    # or a REST client triggered it
    asyncio.get_running_loop().create_task(manager.broadcast(snapshot))
    return snapshot


@app.post("/api/speed")
async def post_speed(request: SpeedRequest) -> dict:
    manager.delay = 1.0 / request.gps
    snapshot = manager.snapshot()
    asyncio.get_running_loop().create_task(manager.broadcast(snapshot))
    return snapshot


# --------------------------------------------------------------------------
# WebSocket + static dashboard
# --------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    manager.clients.add(websocket)
    try:
        await websocket.send_text(json.dumps(manager.snapshot(), ensure_ascii=False))
        while True:
            # Client messages are ignored; the socket is receive-only by design
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        manager.clients.discard(websocket)


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    # no-store: the page must always match the static assets beside it
    return FileResponse(ROOT / "static" / "index.html",
                        headers={"Cache-Control": "no-store"})
