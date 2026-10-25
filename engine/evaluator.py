"""Numeric evaluation of formula trees on n = 1..len(seq).

Guards are part of the contract: division by zero, overflow, complex
results and NaN never raise — the offender evaluates to +inf, which the
fitness layer maps to the worst possible score. Without this a single
bad candidate would kill a whole generation.
"""

from __future__ import annotations

import math

from .nodes import BinOp, Const, Neg, Node, UnOp, Var

# Anything beyond this magnitude counts as overflow: it already dwarfs
# every sane sequence target and keeps the fitness math in safe ranges.
BIG = 1e100


def _cbrt(x: float) -> float:
    # Real cube root: Python's x ** (1/3) goes complex for x < 0.
    return math.copysign(abs(x) ** (1 / 3), x)


def _fact(x: float) -> float:
    # Only exact non-negative integers up to 170: anything else would
    # produce ValueError, floats pretending to be integers, or huge
    # exact ints that slow evaluation down.
    k = round(x)
    if abs(x - k) > 1e-9 or k < 0 or k > 170:
        raise ValueError("factorial out of domain")
    return float(math.factorial(k))


# Every entry either returns a finite float or raises one of the
# exceptions evaluate() already guards (ValueError / OverflowError /
# ZeroDivisionError / TypeError). Unknown names raise too — a corrupt
# AST must degrade to an invalid candidate, never crash the engine.
UNARY_EVAL = {
    "sqrt": math.sqrt,       # ValueError for negatives
    "cbrt": _cbrt,
    "fact": _fact,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,         # finite but huge near poles; BIG filters
    "ln": math.log,          # ValueError for <= 0
}


def evaluate(node: Node, n: float) -> float:
    try:
        value = _eval(node, n)
    except (ZeroDivisionError, OverflowError, ValueError, TypeError, KeyError):
        return math.inf
    # A fractional power of a negative base returns complex, not an
    # exception — check the type before isfinite, which raises on complex.
    if isinstance(value, complex):
        return math.inf
    if not math.isfinite(value) or abs(value) > BIG:
        return math.inf
    return value


def _eval(node: Node, n: float) -> float:
    if isinstance(node, Const):
        return node.value
    if isinstance(node, Var):
        return n
    if isinstance(node, Neg):
        return -_eval(node.child, n)
    if isinstance(node, UnOp):
        fn = UNARY_EVAL.get(node.name)
        if fn is None:
            raise ValueError(f"unknown function: {node.name!r}")
        return fn(_eval(node.child, n))
    left = _eval(node.left, n)
    right = _eval(node.right, n)
    if node.op == "+":
        return left + right
    if node.op == "-":
        return left - right
    if node.op == "*":
        return left * right
    if node.op == "/":
        return left / right
    if node.op == "mod":
        return left % right
    return left ** right  # ^


def evaluate_all(node: Node, length: int) -> list[float] | None:
    """Values for n = 1..length, or None if the candidate blows up
    anywhere — the caller treats None as an invalid formula."""
    values = [evaluate(node, i) for i in range(1, length + 1)]
    if math.inf in values:
        return None
    return values
