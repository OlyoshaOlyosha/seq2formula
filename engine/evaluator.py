"""Numeric evaluation of formula trees on n = 1..len(seq).

Guards are part of the contract: division by zero, overflow, complex
results and NaN never raise — the offender evaluates to +inf, which the
fitness layer maps to the worst possible score. Without this a single
bad candidate would kill a whole generation.
"""

from __future__ import annotations

import math

from .nodes import BinOp, Const, Neg, Node, Var

# Anything beyond this magnitude counts as overflow: it already dwarfs
# every sane sequence target and keeps the fitness math in safe ranges.
BIG = 1e100


def evaluate(node: Node, n: float) -> float:
    try:
        value = _eval(node, n)
    except (ZeroDivisionError, OverflowError, ValueError, TypeError):
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
    return left ** right  # ^


def evaluate_all(node: Node, length: int) -> list[float] | None:
    """Values for n = 1..length, or None if the candidate blows up
    anywhere — the caller treats None as an invalid formula."""
    values = [evaluate(node, i) for i in range(1, length + 1)]
    if math.inf in values:
        return None
    return values
