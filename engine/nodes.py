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


def const_weight(value: float) -> float:
    """Parsimony cost of a constant. Small ints and halves are atomic
    primitives (they ARE the answer to many puzzles); 1000 is a compound
    value that just happens to have short notation. Cost grows with the
    digit count — a proxy for 'how compound a number is'. This stops
    the search from filling formulas with magic constants that fit by
    coincidence (same reasoning as ^ and mod costing 2 in
    weighted_size: unusual values should earn their place)."""
    if abs(value) < 1:
        return 1.0  # halves are primitives (0.5, 1.5 build rational puzzles)
    return float(len(str(int(abs(value)))))  # 1-9: 1, 10-99: 2, 100-999: 3, 1000: 4


def weighted_size(node: Node) -> float:
    """Complexity of a tree: basic arithmetic is cheap, exotic ops and
    large constants are expensive. Used for parsimony so the search
    prefers 2*n+1 over (n mod 4)! or 1000*n/500 even when both are exact."""
    if isinstance(node, BinOp):
        # Exponentiation (^) is as fundamental as multiplication, so it
        # costs 1.0. Modulo is a "patch" operation, it costs 2.0.
        weight = 2.0 if node.op == "mod" else 1.0
        return weight + weighted_size(node.left) + weighted_size(node.right)
    if isinstance(node, UnOp):
        return 2.0 + weighted_size(node.child)
    if isinstance(node, Neg):
        return 1.0 + weighted_size(node.child)
    if isinstance(node, Const):
        return const_weight(node.value)
    return 1.0  # Var


def depth(node: Node) -> int:
    if isinstance(node, BinOp):
        return 1 + max(depth(node.left), depth(node.right))
    if isinstance(node, (Neg, UnOp)):
        return 1 + depth(node.child)
    return 1


def _extract_factors(node: Node) -> list[Node]:
    """Recursively flatten multiplication chains into a list of factors."""
    if isinstance(node, BinOp) and node.op == "*":
        return _extract_factors(node.left) + _extract_factors(node.right)
    return [node]


def _extract_terms(node: Node, sign: float = 1.0) -> list[tuple[float, Node]]:
    """Recursively flatten +/- chains into a list of (coefficient, term)."""
    if isinstance(node, BinOp) and node.op == "+":
        return _extract_terms(node.left, sign) + _extract_terms(node.right, sign)
    if isinstance(node, BinOp) and node.op == "-":
        return _extract_terms(node.left, sign) + _extract_terms(node.right, -sign)
    if isinstance(node, Neg):
        return _extract_terms(node.child, -sign)
    if isinstance(node, BinOp) and node.op == "*":
        if isinstance(node.left, Const):
            return [(sign * node.left.value, node.right)]
        if isinstance(node, Const) or isinstance(node.right, Const):
            if isinstance(node.right, Const):
                return [(sign * node.right.value, node.left)]
    return [(sign, node)]


def simplify(node: Node, ops: tuple[str, ...] = ("+", "-", "*", "/", "^", "mod")) -> Node:
    """Algebraic simplification: x + x -> 2 * x, x * 1 -> x, etc.
    Makes formulas shorter, prettier, and faster to evaluate."""
    if isinstance(node, BinOp):
        left = simplify(node.left, ops)
        right = simplify(node.right, ops)

        # 1. Flatten and combine additions/subtractions (x + x -> 2x, n + n - 1 -> 2n - 1)
        if node.op in ("+", "-"):
            terms = _extract_terms(BinOp(node.op, left, right))
            combined: dict[Node, float] = {}
            for coeff, term in terms:
                # Round to 6 decimals to prevent float artifacts (0.1 + 0.2 -> 0.3)
                combined[term] = round(combined.get(term, 0.0) + coeff, 6)

            # Remove zeros
            combined = {t: c for t, c in combined.items() if abs(c) > 1e-9}

            if not combined:
                return Const(0.0)

            # Sort: variables first, constants last
            sorted_terms = sorted(combined.items(), key=lambda item: isinstance(item[0], Const))

            result = None
            for term, coeff in sorted_terms:
                sign = 1
                if coeff < 0:
                    sign = -1
                    coeff = abs(coeff)

                if isinstance(term, Const):
                    val = sign * coeff * term.value
                    term_node = Const(val)
                elif abs(coeff - 1.0) < 1e-9:
                    term_node = term
                else:
                    term_node = BinOp("*", Const(coeff), term)

                if result is None:
                    result = term_node if sign == 1 or isinstance(term_node, Const) else Neg(term_node)
                else:
                    if sign == -1:
                        result = BinOp("-", result, term_node)
                    else:
                        result = BinOp("+", result, term_node)
            return result

        # 2. Constant folding
        if isinstance(left, Const) and isinstance(right, Const):
            try:
                if node.op == "*": return Const(left.value * right.value)
                if node.op == "/" and right.value != 0: return Const(left.value / right.value)
                if node.op == "mod" and right.value != 0: return Const(left.value % right.value)
                if node.op == "^":
                    val = left.value ** right.value
                    if isinstance(val, complex) or not math.isfinite(val): return BinOp(node.op, left, right)
                    return Const(val)
            except (OverflowError, ZeroDivisionError, ValueError):
                pass

        # 3. Pull division out of multiplication: (a / b) * c -> (a * c) / b
        # This allows the multiplication rules below to combine the numerators.
        if isinstance(left, BinOp) and left.op == "/":
            return simplify(BinOp("/", BinOp("*", left.left, right), left.right), ops)
        if isinstance(right, BinOp) and right.op == "/":
            return simplify(BinOp("/", BinOp("*", left, right.left), right.right), ops)

        # 4. Flatten and combine multiplications (n * 0.5 * n -> 0.5 * n^2)
        if node.op == "*":
            factors = _extract_factors(BinOp("*", left, right))
            const_val = 1.0
            term_counts = {}
            
            for f in factors:
                if isinstance(f, Const):
                    const_val *= f.value
                else:
                    term_counts[f] = term_counts.get(f, 0) + 1
                    
            if abs(const_val) < 1e-9:
                return Const(0.0)
                
            res_factors = []
            if abs(const_val - 1.0) > 1e-9 or not term_counts:
                res_factors.append(Const(const_val))
                
            for term, count in term_counts.items():
                if count == 1:
                    res_factors.append(term)
                elif "^" in ops:
                    res_factors.append(BinOp("^", term, Const(float(count))))
                else:
                    # If ^ is disabled, just multiply the term by itself
                    res_factors.extend([term] * count)
                    
            res = res_factors[0]
            for f in res_factors[1:]:
                res = BinOp("*", res, f)
            return res
        if node.op == "/":
            if isinstance(right, Const) and right.value == 1: return left
        if node.op == "^":
            if isinstance(right, Const) and right.value == 1: return left
            if isinstance(right, Const) and right.value == 0: return Const(1.0)

        return BinOp(node.op, left, right)
        
    elif isinstance(node, UnOp):
        child = simplify(node.child, ops)
        if node.name == "abs" and isinstance(child, Const):
            return Const(abs(child.value))
        return UnOp(node.name, child)
        
    elif isinstance(node, Neg):
        child = simplify(node.child, ops)
        if isinstance(child, Const):
            return Const(-child.value)
        if isinstance(child, Neg):
            return child.child
        # Fold -1 * x into Neg(x) for cleaner rendering
        if isinstance(child, BinOp) and child.op == "*" and isinstance(child.left, Const):
            return BinOp("*", Const(-child.left.value), child.right)
        if isinstance(child, BinOp) and child.op == "*" and isinstance(child.right, Const):
            return BinOp("*", Const(-child.right.value), child.left)
        return Neg(child)
        
    return node


def _num_text(value: float) -> str:
    if abs(value - PI) < 1e-12:
        return "π"
    if abs(value - E) < 1e-12:
        return "e"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    # Round to 6 significant digits to avoid float artifacts (0.666666...)
    return f"{value:.6g}"


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
