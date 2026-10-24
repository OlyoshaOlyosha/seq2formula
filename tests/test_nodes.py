"""Tests for the AST: rendering, serialization, structural metrics."""

import pytest

from engine.nodes import (BinOp, Const, Neg, Var, VAR, depth, from_dict,
                          size, to_dict, to_text)


def C(v):
    return Const(float(v))


def B(op, left, right):
    return BinOp(op, left, right)


@pytest.mark.parametrize("node, expected", [
    (C(2), "2"),
    (C(2.5), "2.5"),
    (VAR, "n"),
    (Neg(VAR), "-n"),
    (B("+", VAR, C(3)), "n + 3"),
    (B("+", VAR, Neg(C(3))), "n - 3"),
    (B("-", C(3), Neg(C(2))), "3 + 2"),
    (B("*", C(2), VAR), "2 * n"),
    (B("/", B("+", VAR, C(3)), C(2)), "(n + 3) / 2"),
    (B("+", B("*", VAR, C(2)), C(1)), "n * 2 + 1"),
], ids=["const", "frac-const", "var", "neg", "add", "fold-neg-add",
        "fold-double-neg", "mul", "div-parens", "prec-mul-over-add"])
def test_to_text_known_forms(node, expected):
    assert to_text(node) == expected


def test_power_rendering():
    assert to_text(B("^", C(2), C(3))) == "2 ^ 3"


def test_neg_binds_looser_than_power():
    node = Neg(B("^", C(2), C(2)))
    # -2^2 reads as -(2^2), the standard math convention
    assert to_text(node) == "-2 ^ 2"


def test_size_and_depth():
    tree = B("/", B("+", VAR, C(3)), C(2))  # (n+3)/2
    assert size(tree) == 5
    assert depth(tree) == 3
    assert size(VAR) == 1 and depth(VAR) == 1


@pytest.mark.parametrize("node", [
    C(2.5),
    VAR,
    Neg(B("*", VAR, C(3))),
    B("/", B("-", B("^", VAR, C(2)), C(1)), C(2)),
], ids=["const", "var", "neg", "nested"])
def test_dict_roundtrip(node):
    assert from_dict(to_dict(node)) == node


def test_from_dict_rejects_unknown_kind():
    with pytest.raises(ValueError):
        from_dict({"t": "@", "l": {"t": "n"}, "r": {"t": "n"}})


def test_from_dict_rejects_bad_const():
    with pytest.raises(ValueError):
        from_dict({"t": "c", "v": "big"})


def test_from_dict_rejects_non_dict():
    with pytest.raises(ValueError):
        from_dict(["n"])


def test_from_dict_depth_cap():
    # 70 nested negations must be rejected instead of blowing the stack
    data = {"t": "n"}
    for _ in range(70):
        data = {"t": "~", "x": data}
    with pytest.raises(ValueError):
        from_dict(data)


def test_frozen_immutability():
    with pytest.raises(Exception):
        Const(1.0).value = 2.0
