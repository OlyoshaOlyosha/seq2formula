"""Neuro engine (stage 2): a small policy network builds formulas
token-by-token and learns from the SAME fitness the GP engine uses.

How it works, in one paragraph: at every decision point of an expression
slot (root / operand / function argument) the network sees the slot's
context (depth budget left, slot kind, previously chosen token) plus
features of the target sequence, and outputs a distribution over the
grammar's token vocabulary. Sampling tokens until the root slot resolves
yields a complete AST — the same grammar the GP engine uses, so every
candidate is scored by score_candidate unchanged. One step() = one
generation = one REINFORCE update over a batch of sampled formulas:
advantage = reward - running baseline, gradient = advantage-weighted
log-probability plus an entropy bonus against premature collapse.

Why REINFORCE (and not DQN / imitation): the action space is a tiny
~40-token vocabulary and the episode is a handful of decisions, so a
vanilla policy gradient with a baseline is the simplest learner that
works; DQN adds machinery the problem does not need, and imitation of GP
solutions would contradict the project's point (the engine must find
formulas itself). Everything runs on numpy — no ML dependency.

Design note on `^`: the grammar forbids n^n, so the vocabulary has two
dedicated power tokens, n^k and k^n, each of which opens exactly one
expression slot plus one constant-only decision for the exponent/base.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np

from .fitness import INVALID_FITNESS, Score, score_candidate
from .gp import Best, GenerationResult
from .nodes import BinOp, Const, Neg, Node, UnOp, Var

# Vocabulary: constants get their own tokens (the network must learn WHICH
# number fits the puzzle — 2 for 2 4 8, bias 5 for 6 7 8), so the token set
# mirrors the grammar's draw buckets: small ints, halves, one large.
DEFAULT_CONSTS = (
    -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    7.0, 8.0, 9.0, 10.0, 0.5, 1.5, 100.0,
)
EXP_BASE_CONSTS = (-3.0, -2.0, -1.0, 2.0, 3.0)  # exponent / base for ^ forms
POW_NK, POW_KN = "^nk", "^kn"
NEG_TOKEN = "neg"  # noqa: S105  (grammar token, not a credential)

# Slot kinds fed to the policy (one-hot): root of the tree, left/right
# operand of a binary op, argument of a unary function or neg, and the
# constant-only exponent/base choice inside a ^ form.
SLOT_ROOT, SLOT_LEFT, SLOT_RIGHT, SLOT_ARG, SLOT_POW = range(5)

# Stabilizers: advantages are centred (running baseline) and scaled by
# the batch std — reward spread is tiny early on, so raw advantages make
# vanishingly small gradients; one lucky exact hit early on must also not
# blow the policy up.
# Anti-collapse ("shock"): the policy can converge to a confidently WRONG
# distribution (entropy -> 0) and never leave a local optimum like 1.5n
# for 2n. The GP engine shocks its population; the same idea here is
# Gaussian noise on the policy weights after this many generations
# without a new best fitness. 0 disables.
STAGNATION_WINDOW = 30
SHOCK_SIGMA = 0.05

# Epsilon-soft sampling: every legal token keeps at least an EPS share of
# the probability mass. Deceptive landscapes (1.5n is a real local optimum
# for 2n-1 — the intermediate 2n scores WORSE) need the exact formula to
# stay reachable by chance forever, like the GP engine's random immigrants.
SOFT_EPS = 0.1

# Cross-entropy method: the update is maximum-likelihood on the top
# ELITE_FRAC of each batch instead of advantage-weighting every episode.
# REINFORCE's tiny per-decision gradients crawl along the deceptive
# landscape; a lucky exact formula joins the elite instantly (fitness
# 0.995 vs 0.828) and the policy snaps to it wholesale next update.
ELITE_FRAC = 0.25

# Reward shaping via first differences: CEM/REINFORCE need a gradient path
# to follow, but some landscapes are deceptive — for 2n-1 the intermediate
# 2n scores WORSE than the attractor 1.5n under the plain error metric.
# Candidates whose first differences match the target's get a bonus, which
# lifts every correct-slope formula above the attractor and turns the last
# constant (the offset) into a small hill instead of a cliff. Weight >= 0.7
# is required for 2n to actually outrank 1.5n (fitness gap 0.175 over the
# shaping gap 0.25 * W); 1.0 keeps margin. Purely a reward-side term: the
# solve criterion stays score_candidate's exact hit.
DIFF_W = 1.0


@dataclass(slots=True)
class NeuroConfig:
    # Defaults tuned on the T1 set (9 puzzles x 3 seeds): pop 256 with
    # lr 0.02 / beta 0.02 solved everything except 2n+-1 variants (which
    # still solve on most seeds); lower pop or higher lr collapsed into
    # attractors like 1.5n for 2n-1.
    population: int = 256       # formulas sampled per step (= batch size)
    max_depth: int = 6
    lr: float = 0.02            # plain SGD step
    entropy_beta: float = 0.02  # bonus against premature determinism
    baseline_decay: float = 0.9  # running-mean momentum for the baseline
    seed: int = 0
    allowed_ops: tuple[str, ...] = ("+", "-", "*", "/", "^", "mod")
    allowed_funcs: tuple[str, ...] = ()
    allowed_consts: tuple[float, ...] = DEFAULT_CONSTS
    parsimony: float = 0.001


class _Policy:
    """3-layer MLP: context -> hidden -> hidden -> token logits.

    Inputs are plain concatenations of one-hots (puzzle features, depth
    budget, slot kind, previous token), so the first weight matrix doubles
    as the token embedding — no separate embedding table needed.
    """

    def __init__(self, in_dim: int, hidden: int, vocab: int,
                 rng: np.random.Generator):
        # Flat init keeps early logits near-uniform => high-entropy
        # exploration at the start of training.
        self.w1 = rng.normal(0, 1 / math.sqrt(in_dim), (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1 / math.sqrt(hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.w3 = rng.normal(0, 1 / math.sqrt(hidden), (hidden, vocab))
        self.b3 = np.zeros(vocab)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """x: (N, in_dim) batch of decision contexts. Returns softmax
        probabilities plus the cache needed for the backward pass."""
        z1 = x @ self.w1 + self.b1
        h1 = np.tanh(z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.tanh(z2)
        z3 = h2 @ self.w3 + self.b3
        z3 -= z3.max(axis=1, keepdims=True)  # stable softmax
        p = np.exp(z3)
        p /= p.sum(axis=1, keepdims=True)
        return p, {"x": x, "h1": h1, "h2": h2}

    def backward(self, cache: dict, dz3: np.ndarray) -> dict[str, np.ndarray]:
        """dz3: (N, vocab) gradient at the softmax input, already averaged
        upstream."""
        h1, h2, x = cache["h1"], cache["h2"], cache["x"]
        gw3 = h2.T @ dz3
        gb3 = dz3.sum(axis=0)
        dh2 = dz3 @ self.w3.T
        dz2 = dh2 * (1 - h2 ** 2)
        gw2 = h1.T @ dz2
        gb2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.w2.T
        dz1 = dh1 * (1 - h1 ** 2)
        return {"w1": x.T @ dz1, "b1": dz1.sum(axis=0),
                "w2": gw2, "b2": gb2, "w3": gw3, "b3": gb3}

    def sgd(self, grads: dict[str, np.ndarray], lr: float) -> None:
        for name, g in grads.items():
            setattr(self, name, getattr(self, name) - lr * g)


class NeuroEngine:
    """Same external protocol as GPEngine: step() -> GenerationResult,
    solved / best_ever / generation / cfg.seed — the server cannot tell
    the engines apart."""

    def __init__(self, seq: list[float], config: NeuroConfig | None = None):
        self.seq = [float(v) for v in seq]
        self.cfg = config or NeuroConfig()
        self.rng = random.Random(self.cfg.seed)
        self.nprng = np.random.default_rng(self.cfg.seed)

        # Token vocabulary: fixed order = token ids. Power tokens encode
        # the n^n ban structurally; specials (π/e) reuse the const slots.
        self.consts = tuple(dict.fromkeys(self.cfg.allowed_consts))
        self.funcs = tuple(self.cfg.allowed_funcs)
        self.terminal_tokens = ["var"] + [f"c{c:g}" for c in self.consts]
        self.op_tokens = [o for o in self.cfg.allowed_ops if o != "^"]
        self.vocab = (self.terminal_tokens + self.op_tokens
                      + list(self.funcs) + [NEG_TOKEN, POW_NK, POW_KN])
        self.tok_id = {t: i for i, t in enumerate(self.vocab)}
        self.const_ids = [self.tok_id[f"c{c:g}"] for c in self.consts]
        # exponent/base choice must stay inside the safe ^ range
        self.exp_ids = [self.tok_id[f"c{c:g}"] for c in EXP_BASE_CONSTS
                        if f"c{c:g}" in self.tok_id]

        # Puzzle features are constant across the run: length, the first
        # terms and their first differences, all normalized by the max
        # magnitude. A constant difference IS the linear rule (2n-1 ->
        # diffs == 2), so this is the feature that lets the policy pick
        # slope/offset constants directly.
        m = max(1.0, max(abs(v) for v in self.seq))
        head = [v / m for v in self.seq[:6]]
        diffs = [(b - a) / m
                 for a, b in zip(self.seq, self.seq[1:], strict=False)][:5]
        self.puzzle_feats = np.array([len(self.seq) / 12.0, *head, *diffs],
                                     dtype=float)
        self.in_dim = len(self.puzzle_feats) + (self.cfg.max_depth + 1) \
            + 5 + len(self.vocab)

        self.policy = _Policy(self.in_dim, 48, len(self.vocab), self.nprng)
        self.generation = 0
        self.solved = False
        self.best_ever: Best | None = None
        self.baseline = 0.0
        self.stagnation = 0  # generations since the last new best fitness

    # ---- episode generation (policy rollout) ----

    def _legal_mask(self, budget: int, const_only: bool) -> np.ndarray:
        mask = np.zeros(len(self.vocab))
        if const_only:  # the exponent/base decision of a ^ form
            for i in self.exp_ids:
                mask[i] = 1.0
            return mask
        if budget <= 1:  # no room for another internal node: terminal only
            mask[:len(self.terminal_tokens)] = 1.0
            return mask
        mask[:len(self.terminal_tokens)] = 1.0
        for op in self.op_tokens:
            mask[self.tok_id[op]] = 1.0
        if budget >= 2:
            for f in self.funcs:
                mask[self.tok_id[f]] = 1.0
            mask[self.tok_id[NEG_TOKEN]] = 1.0
            if "^" in self.cfg.allowed_ops:
                mask[self.tok_id[POW_NK]] = 1.0
                mask[self.tok_id[POW_KN]] = 1.0
        return mask

    def _state_vec(self, budget: int, slot: int, last_id: int) -> np.ndarray:
        depth_oh = np.zeros(self.cfg.max_depth + 1)
        depth_oh[min(budget, self.cfg.max_depth)] = 1.0
        slot_oh = np.zeros(5)
        slot_oh[slot] = 1.0
        last_oh = np.zeros(len(self.vocab))
        last_oh[last_id] = 1.0
        return np.concatenate([self.puzzle_feats, depth_oh, slot_oh, last_oh])

    def _rollout_slot(self, budget: int, slot: int, trace: list,
                      last_id: int) -> Node:
        """Fill one expression slot: sample a token, recurse per grammar.
        Every (state, action) pair lands in `trace` for the update."""
        aid = self._sample(self._state_vec(budget, slot, last_id),
                           self._legal_mask(budget, const_only=False), trace)
        tok = self.vocab[aid]

        if tok == "var":
            return Var()
        if tok.startswith("c"):
            return Const(float(tok[1:]))
        if tok == NEG_TOKEN:
            return Neg(self._rollout_slot(budget - 1, SLOT_ARG, trace, aid))
        if tok in self.funcs:
            return UnOp(tok, self._rollout_slot(budget - 1, SLOT_ARG,
                                                trace, aid))
        if tok in (POW_NK, POW_KN):
            # constant side first: one const-only decision, then the
            # single open slot (n^k -> slot is the base, k^n -> exponent)
            cid = self._sample(self._state_vec(budget - 1, SLOT_POW, aid),
                               self._legal_mask(budget - 1, const_only=True),
                               trace)
            const = Const(float(self.vocab[cid][1:]))
            sub = self._rollout_slot(budget - 1, SLOT_RIGHT, trace, cid)
            if tok == POW_NK:
                return BinOp("^", sub, const)
            return BinOp("^", const, sub)
        # plain binary op: two expression slots, left then right
        left = self._rollout_slot(budget - 1, SLOT_LEFT, trace, aid)
        right = self._rollout_slot(budget - 1, SLOT_RIGHT, trace, aid)
        return BinOp(tok, left, right)

    def _sample(self, state: np.ndarray, mask: np.ndarray,
                trace: list) -> int:
        """One policy decision: forward pass on the single state, mask
        illegal tokens, sample by cumulative probability."""
        p, _ = self.policy.forward(state[None, :])
        probs = p[0] * mask
        total = probs.sum()
        if total <= 1e-9:
            # defensive: masks are built to be non-empty, but never crash
            probs = mask if mask.any() else None
            if probs is None:
                probs = np.zeros(len(self.vocab))
                probs[:len(self.terminal_tokens)] = 1.0
            total = probs.sum()
        probs = probs / total
        # epsilon-soft: mix in the uniform distribution over legal tokens
        legal = int((mask > 0).sum()) or 1
        probs = (1 - SOFT_EPS) * probs + SOFT_EPS * mask / legal
        probs = probs / probs.sum()
        r = self.rng.random()
        acc = 0.0
        aid = len(self.vocab) - 1
        for i, pr in enumerate(probs):
            acc += pr
            if r <= acc:
                aid = i
                break
        trace.append((state, aid))
        return aid

    # ---- learning ----

    def _update(self, traces: list[list], rewards: list[float],
                vals: list) -> dict:
        """One cross-entropy update: supervised MLE on the elite
        episodes' decisions. All elite decisions go through the network
        as one matrix; a small entropy bonus keeps the sampled policy
        from collapsing onto the elite while SOFT_EPS guards coverage.
        The elite is deduped by the values its formulas produce (same
        trick as the GP engine's value-based dedupe): without it a single
        mode fills every elite slot and structurally different families
        (n*n vs -c chains) never get gradient."""
        batch_r = np.array(rewards)
        k = max(2, int(len(rewards) * ELITE_FRAC))
        elite: list[int] = []
        seen: set = set()
        for i in sorted(range(len(rewards)), key=lambda i: rewards[i],
                        reverse=True):
            if len(elite) >= k:
                break
            v = vals[i]
            if v is None or v in seen:
                continue  # invalid or a duplicate of an existing mode
            seen.add(v)
            elite.append(i)

        # Recompute forward on full batches (the rollouts ran row-by-row),
        # then re-apply the epsilon-soft mixing so the gradient is the
        # log-prob UNDER THE SAME distribution the action was sampled
        # from (the depth/slot one-hots inside each state row let us
        # rebuild the legal mask without storing it in the trace).
        x = np.stack([s for tr in traces for (s, _) in tr])
        aids = [a for tr in traces for (_, a) in tr]
        p, cache = self.policy.forward(x)
        d0 = len(self.puzzle_feats)
        depths = x[:, d0:d0 + self.cfg.max_depth + 1].argmax(axis=1)
        slots = x[:, d0 + self.cfg.max_depth + 1:d0 + self.cfg.max_depth + 6]
        slots = slots.argmax(axis=1)
        masks = np.stack([self._legal_mask(int(b), int(s) == SLOT_POW)
                          for b, s in zip(depths, slots, strict=True)])
        legal = masks.sum(axis=1, keepdims=True)
        p = (1 - SOFT_EPS) * p + SOFT_EPS * masks / legal
        p /= p.sum(axis=1, keepdims=True)
        logp = np.log(p + 1e-9)
        h_rows = -(p * logp).sum(axis=1)  # per-decision entropy
        entropy = float(h_rows.mean())

        # elite decisions = the rows belonging to elite episodes
        start = 0
        elite_rows: list[int] = []
        elite_set = set(elite)
        for i, tr in enumerate(traces):
            if i in elite_set:
                elite_rows.extend(range(start, start + len(tr)))
            start += len(tr)
        rows = np.array(elite_rows, dtype=int)
        n = max(1, len(rows))
        # MCE loss on elite actions: d(-log p_a)/dz = p - onehot(a);
        # a small entropy bonus pushes the REST of the distribution away
        # from collapse (maximize H => +beta * dH/dz)
        dz3 = np.zeros_like(p)
        dz3[rows] = p[rows]
        dz3[rows, [aids[r] for r in rows]] -= 1.0
        dz3 /= n
        dz3 += (self.cfg.entropy_beta / len(p)) * p * (logp + h_rows[:, None])
        grads = self.policy.backward(cache, dz3)
        self.policy.sgd(grads, self.cfg.lr)
        self.baseline = (self.cfg.baseline_decay * self.baseline
                         + (1 - self.cfg.baseline_decay) * batch_r.mean())

        return {"reward": float(batch_r.mean()),
                "best_reward": float(batch_r.max()),
                "entropy": entropy,
                "baseline": float(self.baseline),
                "grad_norm": float(math.sqrt(
                    sum(float(np.sum(g * g)) for g in grads.values())))}

    # ---- protocol ----

    def _reward(self, sc: Score) -> float:
        """Training reward: fitness plus the first-difference match bonus
        (shaping only — solve status comes from score_candidate alone)."""
        if sc.fitness == INVALID_FITNESS:
            return sc.fitness
        diffs = [b - a for a, b in zip(self.seq, self.seq[1:], strict=False)]
        got = [b - a for a, b in zip(sc.values, sc.values[1:], strict=False)]
        d_err = sum(min(1.0, abs(g - t) / max(1.0, abs(t)))
                    for g, t in zip(got, diffs, strict=True)) / len(diffs)
        return sc.fitness + DIFF_W * (1.0 - d_err)

    def step(self) -> GenerationResult:
        t0 = time.perf_counter()
        self.generation += 1
        scored: list[tuple[Node, Score]] = []
        traces: list[list] = []
        for _ in range(self.cfg.population):
            trace: list = []
            node = self._rollout_slot(self.cfg.max_depth, SLOT_ROOT,
                                      trace, last_id=0)
            scored.append((node, score_candidate(node, self.seq,
                                                 self.cfg.parsimony)))
            traces.append(trace)
        stats = self._update(traces, [self._reward(sc) for _, sc in scored],
                             [tuple(round(v, 6) for v in sc.values)
                              if sc.values is not None else None
                              for _, sc in scored])

        valid = [s for s in scored if s[1].fitness != INVALID_FITNESS]
        valid.sort(key=lambda s: s[1].fitness, reverse=True)
        stats["valid_frac"] = len(valid) / len(scored)

        # protocol bookkeeping, same semantics as the GP engine
        new_best = bool(valid) and (
            self.best_ever is None
            or valid[0][1].fitness > self.best_ever.score.fitness)
        solved_now = bool(valid) and valid[0][1].exact
        self.solved = self.solved or solved_now
        if valid and (self.best_ever is None
                      or valid[0][1].fitness > self.best_ever.score.fitness):
            self.best_ever = Best(valid[0][0], valid[0][1])
        if self.best_ever is not None:
            stats["best_fitness"] = self.best_ever.score.fitness

        # stagnation shock: perturb the policy weights and reset the
        # baseline so stale reward levels stop anchoring the advantages
        self.stagnation = 0 if new_best else self.stagnation + 1
        shock = (STAGNATION_WINDOW > 0 and not self.solved
                 and self.stagnation >= STAGNATION_WINDOW)
        if shock:
            for name in ("w1", "w2", "w3"):
                w = getattr(self.policy, name)
                setattr(self.policy, name,
                        w + self.nprng.normal(0, SHOCK_SIGMA, w.shape))
            self.baseline = stats["reward"]
            self.stagnation = 0
        stats["shock"] = shock

        top = [b.payload() for _, b in
               ((n, Best(n, sc)) for n, sc in valid[:5])]
        if valid:
            best = Best(*valid[0])
        elif self.best_ever is not None:
            best = self.best_ever
        else:
            best = Best(Const(0.0), Score(INVALID_FITNESS, math.inf, False,
                                          None, None, 1))
        return GenerationResult(
            gen=self.generation, best=best, new_best=new_best,
            solved_now=solved_now, solved=self.solved, shock=shock,
            shock_window=0, top=top, stats=stats,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            population=self.cfg.population,
        )

    def done(self) -> bool:
        return self.solved

    def payload(self) -> dict:  # mirror GPEngine.payload shape
        return {"mode": "neuro", "generation": self.generation}
