"""Tests for the AST: rendering, serialization, structural metrics."""

from dataclasses import FrozenInstanceError

import pytest

from engine.nodes import VAR, BinOp, Const, Neg, depth, from_dict, size, to_dict, to_text


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
    with pytest.raises(FrozenInstanceError):
        Const(1.0).value = 2.0


def test_simplify_division_in_multiplication():
    from engine.nodes import simplify, to_text
    # (n / 2) * n should become n^2 / 2
    tree = B("*", B("/", VAR, C(2)), VAR)
    simplified = simplify(tree)
    text = to_text(simplified)
    assert text in ("n ^ 2 / 2", "n^2 / 2", "1 / 2 * n ^ 2")
    assert simplified.op == "/"
    assert simplified.left.op == "^"


def test_simplify_does_not_bloat_terms():
    from engine.nodes import simplify, size, to_text
    # 7 - n should stay 7 - n (size 3), not become -n + 7 (size 4)
    tree1 = B("-", C(7), VAR)
    sim1 = simplify(tree1)
    assert to_text(sim1) == "7 - n"
    assert size(sim1) == 3

    # -n + 7 should become 7 - n (size 3)
    tree2 = B("+", Neg(VAR), C(7))
    sim2 = simplify(tree2)
    assert to_text(sim2) == "7 - n"
    assert size(sim2) == 3

    # n - 7 should stay n - 7
    tree3 = B("-", VAR, C(7))
    sim3 = simplify(tree3)
    assert to_text(sim3) == "n - 7"
    assert size(sim3) == 3


def test_weighted_size_penalizes_large_constants():
    from engine.nodes import weighted_size
    small = B("*", C(2.0), VAR)          # 2*n
    large = B("*", C(1000.0), VAR)       # 1000*n
    assert weighted_size(small) == 3.0   # 1(op) + 1(const) + 1(n)
    assert weighted_size(large) == 6.0   # 1(op) + 4(const) + 1(n)
    # fractions stay primitive: no penalty for 0.5
    half = B("*", C(0.5), VAR)
    assert weighted_size(half) == 3.0
