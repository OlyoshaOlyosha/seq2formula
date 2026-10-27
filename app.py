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
import random
import re
import socket
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import UNARY_FUNCS, GPConfig, GPEngine, to_dict, to_text
from engine.grammar import BINARY_OPS

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
DEFAULT_GPS = 100.0

# Engine knobs the Настройки tab can change; a fresh store starts here.
# Population/parsimony/max_generations default a RUN when the request does
# not carry them; the rest configure GPConfig directly.
DEFAULT_SETTINGS = {
    "population": 500,
    "max_generations": 10000,
    "speed_max": 100,         # upper bound of the dashboard speed slider
    "parsimony": 0.003,
    "max_depth": 6,
    "tournament": 3,
    "p_crossover": 1.0,       # sweep.py 2026-10-25: 40/41 puzzles vs 39
    "p_mutate": 0.25,
    "elitism": 2,
    "shrink_generations": 0,
    "shrink_parsimony": 0.02,
    "autopilot_gens": 500,
    "notify_flash": True,
    "notify_sound": True,
    "notify_volume": 0.1,
}


def load_settings() -> dict:
    stored = _load_json(SETTINGS_FILE, {})
    settings = dict(DEFAULT_SETTINGS)
    settings.update({k: v for k, v in stored.items() if k in DEFAULT_SETTINGS})
    return settings

app = FastAPI(title="seq2formula")


@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    # Local single-user app: never cache anything to avoid stale static files.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


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
    reference: str | None = None  # user-provided reference formula
    # empty = fresh random seed per run; an integer = deterministic replay
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class AutopilotItem(BaseModel):
    puzzle_id: str | None = None
    terms: list[float] | None = None
    reference: str | None = None


class AutopilotRequest(BaseModel):
    items: list[AutopilotItem]
    gens: int = Field(ge=1, le=10000)
    ops: list[str] | None = None


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
    autopilot_gens: int = Field(ge=1, le=10000)
    notify_flash: bool = True
    notify_sound: bool = True
    notify_volume: float = Field(ge=0.0, le=1.0)


class ControlRequest(BaseModel):
    action: str = Field(pattern="^(pause|resume|step|stop|continue)$")


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
        self.config_summary: dict = {}
        self.continue_mode = False
        self.custom_reference: str | None = None
        self.clients: set[WebSocket] = set()
        
        # Autopilot queue state
        self.is_autopilot = False
        self.autopilot_queue: list[AutopilotItem] = []
        self.autopilot_gens = 0
        self.autopilot_ops = None

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
        # Always hand the engine a concrete integer seed: an empty request
        # seed means "fresh random", and we want the value that was actually
        # used stored in history so the run can be replayed later.
        seed = (request.seed if request.seed is not None
                else random.randrange(2**31))
        config = GPConfig(
            seed=seed,
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
        # In autopilot mode, continue_mode is forced on so we don't stop
        # on a solution until the generation budget is exhausted.
        self.continue_mode = self.is_autopilot
        self.started_at = time.perf_counter()
        self.solved_wall_time = None
        self.puzzle = puzzle
        self.terms = terms
        self.last_gen = None
        self.finish_reason = None
        self.custom_reference = request.reference
        # remembered so the history entry can show how the run was tuned
        self.config_summary = {
            "seed": config.seed,
            "population": config.population,
            "parsimony": config.parsimony,
            "p_mutate": config.p_mutate,
            "p_crossover": config.p_crossover,
            "tournament": config.tournament,
            "elitism": config.elitism,
            "speed": request.speed,
        }
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
            self._save_custom_best()
        elif action == "resume" and self.status in ("paused", "running"):
            self.status = "running"
        elif action == "continue":
            # Keep exploring after the run finished: more exact variants
            # (or even shorter ones) may still turn up. Runs until Стоп.
            if self.status != "finished" or not self.engine.solved:
                raise HTTPException(409, "continue needs a finished solved run")
            self.continue_mode = True
            self.status = "running"
            # the paced loop returned when the run finalized — start a new one
            if self.task is None or self.task.done():
                self.task = asyncio.get_running_loop().create_task(self._loop())
        elif action == "step":
            if self.status == "finished":
                raise HTTPException(409, "run already finished")
            self.status = "paused"
            self.step_once = True
        elif action == "stop":
            self.is_autopilot = False
            self.autopilot_queue.clear()
            self.stop(reason="stopped_by_user")
            return self.snapshot()
        return self.snapshot()

    def start_autopilot(self, req: AutopilotRequest) -> dict:
        self.is_autopilot = True
        self.autopilot_queue = list(req.items)
        self.autopilot_gens = req.gens
        self.autopilot_ops = req.ops
        return self._next_autopilot_run()

    def _next_autopilot_run(self) -> dict:
        if not self.autopilot_queue:
            self.is_autopilot = False
            self.status = "finished"
            self.finish_reason = "autopilot_done"
            return self.snapshot()

        item = self.autopilot_queue.pop(0)
        req = RunRequest(
            puzzle_id=item.puzzle_id,
            terms=item.terms,
            reference=item.reference,
            ops=self.autopilot_ops,
            speed=240, # max speed for autopilot
            max_generations=self.autopilot_gens,
        )
        return self.start(req)

    def _save_custom_best(self) -> None:
        """Save the current best formula to history even if unsolved."""
        if not self.engine or not self.engine.best_ever or not self.terms:
            return
        best = self.engine.best_ever
        
        # Save any progress that isn't complete garbage (error < 99%)
        if best.score.error > 0.99:
            return

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
            "error": round(best.score.error, 6),
            "generations": self.engine.generation,
            "time_s": round(time.perf_counter() - self.started_at, 3),
            "settings": self.config_summary,
            "reference": self.custom_reference,
            "seed": self.engine.cfg.seed,
            "solved": False,
        }
        history = _load_json(HISTORY_FILE, [])
        
        # Don't save if we already have a better (lower error) unsolved entry
        existing = [h for h in history if h["seq_key"] == entry["seq_key"] and not h["solved"]]
        if existing:
            min_error = min(h.get("error", 1.0) for h in existing)
            if entry["error"] >= min_error:
                return

        history.append(entry)
        _save_json(HISTORY_FILE, history[-HISTORY_CAP:])

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
            "seed": self.engine.cfg.seed if self.engine else None,
            "sequence": self.terms,
            "puzzle": ({k: self.puzzle[k] for k in ("id", "tier", "name")}
                       if self.puzzle else None),
            "generation": self.engine.generation if self.engine else 0,
            "population": self.engine.cfg.population if self.engine else 0,
            "delay": self.delay,
            "best": self.last_gen["best"] if self.last_gen else None,
            "solved": bool(self.engine and self.engine.solved),
            "finish_reason": self.finish_reason,
            "last_gen": self.last_gen,
            "is_autopilot": self.is_autopilot,
        }

    # -- the paced loop ------------------------------------------------------

    async def _loop(self) -> None:
        # Pace by absolute deadlines: asyncio.sleep on Windows can wake
        # early, so a relative "sleep the remainder" drifts fast and the
        # run overshoots the slider (observed 46/s at 30). A step must
        # not START before the previous period has fully elapsed.
        next_gen: float | None = None
        try:
            while True:
                wants_step = (self.status == "running"
                              or (self.step_once and self.status == "paused"))
                self.step_once = False
                if wants_step and self.engine is not None:
                    loop = asyncio.get_event_loop()
                    if next_gen is not None:
                        behind = next_gen - time.perf_counter()
                        if behind > 0:
                            await asyncio.sleep(behind)
                        elif -behind > self.delay * 3:
                            next_gen = None  # resync after pause/continue
                    if next_gen is None:
                        next_gen = time.perf_counter() + self.delay
                    else:
                        next_gen += self.delay
                    result = await loop.run_in_executor(None, self.engine.step)
                    self.last_gen = result.payload()
                    await self.broadcast(self.last_gen)
                    if result.solved_now:
                        self.solved_wall_time = time.perf_counter() - self.started_at
                    if self.engine.done and not self.continue_mode:
                        await self._finalize()
                        return
                    if self.continue_mode and self.is_autopilot and self.engine.generation >= self.autopilot_gens:
                        await self._finalize()
                        return
                else:
                    next_gen = None
                    await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass

    async def _finalize(self) -> None:
        """Persist the finished run and announce the outcome."""
        if self.engine is None:
            return
        solved = self.engine.solved
        entry = None
        
        is_autopilot = self.is_autopilot
        
        if not solved:
            self._save_custom_best()
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
                "settings": self.config_summary,
                "reference": self.custom_reference,
                "seed": self.engine.cfg.seed,
                "solved": True,
            }
            history = _load_json(HISTORY_FILE, [])
            history.append(entry)
            _save_json(HISTORY_FILE, history[-HISTORY_CAP:])
            self._save_curiosities()
        self.status = "finished"
        self.finish_reason = ("solved" if solved else
                              self.finish_reason or "max_generations")
        
        # Autopilot: broadcast the current finish, then immediately start the next.
        if is_autopilot and self.autopilot_queue:
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
                "is_autopilot": True,
            })
            # Start the next one in the background
            asyncio.get_running_loop().create_task(self._run_next_autopilot())
            return

        # Normal finish (or autopilot queue exhausted)
        if not self.autopilot_queue:
            self.is_autopilot = False
            
        await self.broadcast({
            "type": "solved" if solved else "finished",
            "reason": self.finish_reason if not is_autopilot else "autopilot_done",
            "formula": entry["formula_ast"] if entry else None,
            "formula_text": entry["formula"] if entry else None,
            "size": entry["size"] if entry else None,
            "generations": entry["generations"] if entry else self.engine.generation,
            "time_s": entry["time_s"] if entry else None,
            "hall_entry": entry,
            "shortest_text": (to_text(self.engine.shortest_exact.node)
                              if self.engine.shortest_exact else None),
            "is_autopilot": self.is_autopilot,
        })

    async def _run_next_autopilot(self) -> None:
        await asyncio.sleep(0.5)  # brief pause so the client can process the finish
        if not self.is_autopilot:
            return
        snapshot = self._next_autopilot_run()
        await self.broadcast(snapshot)

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

def calc_ref_size(text: str) -> float:
    """Port of the JS reference size calculator to guarantee matching weights."""
    if not text:
        return 0.0
    formatted = text.lower()
    formatted = formatted.replace("√", " sqrt ").replace("∛", " cbrt ")
    formatted = formatted.replace("π", " pi ").replace("!", " ! ")
    formatted = re.sub(r'(\d)([a-z(])', r'\1 * \2', formatted)
    formatted = re.sub(r'(\))([a-z0-9(])', r'\1 * \2', formatted)

    tokens = re.findall(r'(sqrt|cbrt|fact|abs|sin|cos|tan|ln|pi|e|\d+(?:\.\d+)?|n|[-+*/^%()!])', formatted)

    size = 0.0
    for t in tokens:
        if t in ("(", ")"):
            continue
        if t in ["sqrt", "cbrt", "fact", "abs", "sin", "cos", "tan", "ln", "!", "%", "mod"]:
            size += 2.0
        elif t in ["+", "-", "*", "/", "^"]:
            size += 1.0
        elif t in ["n", "pi", "e"]:
            size += 1.0
        elif re.match(r'^\d+(\.\d+)?$', t):
            num = float(t)
            if num < 1:
                size += 1.0
            else:
                size += float(len(str(int(abs(num)))))
    return size


@app.get("/api/puzzles")
def get_puzzles() -> dict:
    # Reference formulas are shown up front: single-user local toy, the
    # owner wants the target answer visible while watching the search.
    history = _load_json(HISTORY_FILE, [])
    puzzles = load_puzzles()
    
    # Calculate the best generation count per puzzle from history
    best_gens = {}
    for e in history:
        if e.get("solved"):
            key = e["seq_key"]
            if key not in best_gens or e["generations"] < best_gens[key]:
                best_gens[key] = e["generations"]

    # Find dynamic percentiles for relative difficulty
    solved_counts = sorted(best_gens.values())
    easy_threshold, medium_threshold = None, None
    if len(solved_counts) >= 3:
        easy_threshold = solved_counts[len(solved_counts) // 3]
        medium_threshold = solved_counts[(len(solved_counts) * 2) // 3]

    for p in puzzles:
        ref = p.get("reference", "")
        p["ref_size"] = calc_ref_size(ref)
        
        # Attach difficulty badge relative to the user's own history
        key = _seq_key([float(t) for t in p["terms"]])
        if key in best_gens and easy_threshold is not None:
            gens = best_gens[key]
            if gens <= easy_threshold:
                p["difficulty"] = "easy"
            elif gens <= medium_threshold:
                p["difficulty"] = "medium"
            else:
                p["difficulty"] = "hard"
        elif key in best_gens:
            # Not enough data for percentiles yet, but it is solved
            p["difficulty"] = "medium"
        else:
            p["difficulty"] = "unseen"

    return {"puzzles": puzzles}


@app.get("/api/stats")
def get_stats() -> dict:
    """Library-wide summary derived from history + the puzzle list:
    solved fraction, per-tier aggregates and personal records."""
    history = _load_json(HISTORY_FILE, [])
    puzzles = load_puzzles()
    # best (fewest generations) per solved sequence
    best: dict[str, dict] = {}
    for entry in history:
        if not entry.get("solved"):
            continue
        key = entry["seq_key"]
        current = best.get(key)
        if current is None or entry["generations"] < current["generations"]:
            best[key] = entry
    by_tier = []
    for tier in (1, 2, 3, 4, 5):
        tier_puzzles = [p for p in puzzles if p["tier"] == tier]
        tier_best = []
        for p in tier_puzzles:
            entry = best.get(_seq_key([float(t) for t in p["terms"]]))
            if entry:
                tier_best.append(entry)
        gens = [e["generations"] for e in tier_best]
        by_tier.append({
            "tier": tier,
            "total": len(tier_puzzles),
            "solved": len(tier_best),
            "avg_gens": round(sum(gens) / len(gens), 1) if gens else None,
            "record": (min(tier_best, key=lambda e: e["generations"])
                       if tier_best else None),
        })
    records = sorted((e for e in best.values() if e.get("puzzle")),
                     key=lambda e: e["generations"])[:5]
    clean = [{k: e[k] for k in ("puzzle", "sequence", "formula",
                                "generations", "time_s", "size")}
             for e in records]
    for r, e in zip(clean, records, strict=True):
        # how many solved attempts this sequence has in the history
        r["attempts"] = sum(1 for h in history
                            if h.get("solved") and h["seq_key"] == e["seq_key"])
    return {"total": len(puzzles), "solved": len(best), "by_tier": by_tier,
            "records": clean}


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


@app.post("/api/autopilot")
async def post_autopilot(request: AutopilotRequest) -> dict:
    snapshot = manager.start_autopilot(request)
    asyncio.get_running_loop().create_task(manager.broadcast(snapshot))
    return snapshot


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


def find_free_port(start_port: int = 8002) -> int:
    """Return the first available port, starting from start_port."""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # connect_ex returns 0 if the port is in use
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free ports found in range 8002-8101")


def open_browser(url: str) -> None:
    """Wait a moment for the server to start, then open the browser."""
    time.sleep(1.5)
    webbrowser.open(url)


if __name__ == "__main__":
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    
    print(f"Starting seq2formula on {url}")
    
    # Open the browser in a background thread so it doesn't block uvicorn
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    
    uvicorn.run("app:app", host="127.0.0.1", port=port, log_level="info")
