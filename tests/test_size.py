"""Tests for the weighted parsimony size and constant penalties."""

import pytest
from engine.nodes import (
    BinOp, Const, Neg, Var, VAR, UnOp,
    const_weight, weighted_size
)


def C(v):
    return Const(float(v))


def B(op, left, right):
    return BinOp(op, left, right)


def test_const_weight():
    # Fractions and small ints are primitives
    assert const_weight(0.5) == 1.0
    assert const_weight(-1.5) == 1.0
    assert const_weight(2) == 1.0
    assert const_weight(-5) == 1.0
    # Digit count dictates the weight
    assert const_weight(10) == 2.0
    assert const_weight(-99) == 2.0
    assert const_weight(100) == 3.0
    assert const_weight(1000) == 4.0


def test_weighted_size_basic():
    # 2 * n -> 1 (op) + 1 (const) + 1 (var) = 3
    assert weighted_size(B("*", C(2), VAR)) == 3.0
    # 1000 * n -> 1 (op) + 4 (const) + 1 (var) = 6
    assert weighted_size(B("*", C(1000), VAR)) == 6.0
    # n ^ 2 -> 1 (op) + 1 (var) + 1 (const) = 3 (we reduced ^ weight)
    assert weighted_size(B("^", VAR, C(2))) == 3.0
    # n mod 3 -> 2 (op) + 1 (var) + 1 (const) = 4
    assert weighted_size(B("mod", VAR, C(3))) == 4.0


def test_weighted_size_functions():
    # sin(n) -> 2 (func) + 1 (var) = 3
    assert weighted_size(UnOp("sin", VAR)) == 3.0
    # sqrt(n+1) -> 2 (func) + 1 (op) + 1 (var) + 1 (const) = 5
    assert weighted_size(UnOp("sqrt", B("+", VAR, C(1)))) == 5.0


def test_weighted_size_mixed():
    # (n + 3) / 2 -> 1 (/) + 1 (+) + 1 (n) + 1 (3) + 1 (2) = 5
    tree = B("/", B("+", VAR, C(3)), C(2))
    assert weighted_size(tree) == 5.0

    # 1000 * n + sin(n)
    # 1 (+) + 6 (1000*n) + 3 (sin) = 10
    tree2 = B("+", B("*", C(1000), VAR), UnOp("sin", VAR))
    assert weighted_size(tree2) == 10.0