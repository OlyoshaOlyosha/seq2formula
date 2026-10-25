"""Formula AST: node types, text rendering, JSON serialization.

The AST is the single currency of the system: the GP engine breeds trees,
the evaluator scores them, and the same JSON form crosses the WebSocket
so the page can render formulas with KaTeX. Frozen dataclasses keep trees
immutable and hashable, which the diversity/dedupe logic relies on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Serialization depth cap: from_dict() is fed WebSocket/JSON data, so a
# hostile or corrupt payload must not blow the interpreter stack.
_MAX_JSON_DEPTH = 64

# Grammar v2 unary functions (UnOp nodes). The evaluator holds the
# callables, the grammar draws them, this list is the shared source of
# names — including the JSON tag validation.
UNARY_FUNCS = ("sqrt", "cbrt", "fact", "abs", "sin", "cos", "tan", "ln")

# Special constants drawn from the terminal pool when enabled ("pi",
# "e" picker tokens). Stored as plain Const values — they participate
# in arithmetic, serialization and fitness with no special-casing
# anywhere except text/LaTeX rendering, which recognizes them by value.
PI = math.pi
E = math.e
_SPECIALS = {"pi": PI, "e": E}


def special_value(token: str) -> float | None:
    return _SPECIALS.get(token)


class Node:
    __slots__ = ()


@dataclass(frozen=True)
class Const(Node):
    value: float


@dataclass(frozen=True)
class Var(Node):
    pass


@dataclass(frozen=True)
class Neg(Node):
    child: Node


@dataclass(frozen=True)
class UnOp(Node):
    name: str  # one of UNARY_FUNCS
    child: Node


@dataclass(frozen=True)
class BinOp(Node):
    op: str  # one of + - * / ^ mod
    left: Node
    right: Node


VAR = Var()

_PREC = {"+": 1, "-": 1, "mod": 1, "*": 2, "/": 2, "^": 4}
_NEG_PREC = 3


def size(node: Node) -> int:
    """Number of nodes in the tree (the parsimony measure)."""
    if isinstance(node, BinOp):
        return 1 + size(node.left) + size(node.right)
    if isinstance(node, (Neg, UnOp)):
        return 1 + size(node.child)
    return 1


def depth(node: Node) -> int:
    if isinstance(node, BinOp):
        return 1 + max(depth(node.left), depth(node.right))
    if isinstance(node, (Neg, UnOp)):
        return 1 + depth(node.child)
    return 1


def _num_text(value: float) -> str:
    if abs(value - PI) < 1e-12:
        return "π"
    if abs(value - E) < 1e-12:
        return "e"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def to_text(node: Node) -> str:
    """Infix text with minimal parentheses, by precedence.

    Unary minus binds looser than ^ (-2^n is -(2^n), as in math) and
    tighter than * /. Constant signs fold into the operator (3 + -2 -> 3-2).
    """
    return _fmt(node, 0)


def _fmt(node: Node, parent_prec: int) -> str:
    if isinstance(node, Const):
        return _num_text(node.value)
    if isinstance(node, Var):
        return "n"
    if isinstance(node, Neg):
        return "-" + _fmt(node.child, _NEG_PREC)
    if isinstance(node, UnOp):
        if node.name == "fact":
            return _fact_text(node.child) + "!"
        return f"{node.name}({_fmt(node.child, 0)})"
    # BinOp: fold a negative constant into the operator (n + -3 -> n-3),
    # whether it is stored as Const(-3) or Neg(Const(3))
    neg_value = _neg_const(node.right) if node.op in "+-" else None
    if neg_value is not None:
        op = "-" if node.op == "+" else "+"
        right = _num_text(abs(neg_value))
    else:
        op, right = node.op, _fmt(node.right, _PREC[node.op] + 1)
    left = _fmt(node.left, _PREC[node.op])
    text = f"{left} {op} {right}"
    if _PREC[node.op] < parent_prec:
        return f"({text})"
    return text


def _fact_text(child: Node) -> str:
    """Postfix ! binds tightest: only a bare variable, a non-negative
    constant or an already-parenthesized function call goes without
    parens. Anything else would change the meaning (-3! reads -(3!))."""
    if isinstance(child, Var):
        return _fmt(child, 0)
    if isinstance(child, Const) and child.value >= 0:
        return _fmt(child, 0)
    if isinstance(child, UnOp) and child.name != "fact":
        return _fmt(child, 0)
    if isinstance(child, UnOp):
        return f"({_fmt(child, 0)})"  # n!! would be ambiguous to read
    return f"({_fmt(child, 0)})"


def _neg_const(node: Node) -> float | None:
    if isinstance(node, Const) and node.value < 0:
        return node.value
    if isinstance(node, Neg) and isinstance(node.child, Const):
        return -node.child.value
    return None


def to_dict(node: Node) -> dict:
    """Compact JSON form: c=const, n=var, ~=neg, f=unary function,
    binary op letter / 'mod'."""
    if isinstance(node, Const):
        return {"t": "c", "v": node.value}
    if isinstance(node, Var):
        return {"t": "n"}
    if isinstance(node, Neg):
        return {"t": "~", "x": to_dict(node.child)}
    if isinstance(node, UnOp):
        return {"t": "f", "n": node.name, "x": to_dict(node.child)}
    return {"t": node.op, "l": to_dict(node.left), "r": to_dict(node.right)}


def from_dict(data: dict, _depth: int = 0) -> Node:
    if _depth > _MAX_JSON_DEPTH:
        raise ValueError("formula AST too deep")
    if not isinstance(data, dict) or "t" not in data:
        raise ValueError("bad AST node: expected dict with 't'")
    kind = data["t"]
    if kind == "c":
        value = data.get("v")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("bad constant value")
        return Const(float(value))
    if kind == "n":
        return VAR
    if kind == "~":
        return Neg(from_dict(data["x"], _depth + 1))
    if kind == "f":
        name = data.get("n")
        if name not in UNARY_FUNCS:
            raise ValueError(f"unknown function: {name!r}")
        return UnOp(name, from_dict(data["x"], _depth + 1))
    if kind in _PREC:
        return BinOp(kind, from_dict(data["l"], _depth + 1),
                     from_dict(data["r"], _depth + 1))
    raise ValueError(f"unknown AST node type: {kind!r}")
