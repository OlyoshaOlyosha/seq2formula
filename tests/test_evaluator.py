"""Tests for the guarded evaluator."""

from engine.evaluator import evaluate, evaluate_all
from engine.nodes import VAR, BinOp, Const, Neg


def B(op, left, right):
    return BinOp(op, left, right)


def C(v):
    return Const(float(v))


SEQ = [2.0, 4.0, 8.0, 16.0, 32.0]  # 2^n for n = 1..5


def test_two_pow_n():
    assert evaluate_all(B("^", C(2), VAR), 5) == SEQ


def test_arithmetic_ops():
    assert evaluate(B("+", VAR, C(3)), 4) == 7
    assert evaluate(B("-", VAR, C(3)), 4) == 1
    assert evaluate(B("*", C(2), VAR), 4) == 8
    assert evaluate(B("/", B("+", VAR, C(3)), C(2)), 1) == 2.0


def test_unary_minus():
    assert evaluate(Neg(B("*", C(2), VAR)), 3) == -6


def test_division_by_zero_is_inf_not_exception():
    assert evaluate(B("/", C(1), B("-", VAR, VAR)), 5) == float("inf")


def test_overflow_is_inf_not_exception():
    assert evaluate(B("^", C(10), C(400)), 1) == float("inf")  # 10^400


def test_complex_result_is_inf_not_exception():
    # (-3) ^ (-1/2): a complex value, not an exception — must score invalid
    tree = B("^", C(-3), B("/", C(-1), C(2)))
    assert evaluate(tree, 1) == float("inf")


def test_evaluate_all_none_when_any_term_invalid():
    # 1/(n-3) explodes at n = 3
    tree = B("/", C(1), B("-", VAR, C(3)))
    assert evaluate_all(tree, 5) is None


def test_evaluate_all_ok():
    assert evaluate_all(B("*", C(2), VAR), 3) == [2.0, 4.0, 6.0]
