"""Random formula generation and genetic operators over the AST.

Grammar v1: terminals are n and numeric constants (integers plus a few
halves for the rational puzzles); binary ops are + - * / ^. Power needs
a constant on at least one side: a constant exponent gives n^2-style
polynomials, a constant base gives 2^n-style exponentials; n^n stays
forbidden. Every operator enforces the depth limit and repairs ^ nodes
that lost their constant through a crossover splice.
"""

from __future__ import annotations

import random

from .nodes import BinOp, Const, Neg, Node, Var, VAR, depth, size

BINARY_OPS = ("+", "-", "*", "/", "^")
CONST_RANGE = (-5, 5)
# Half-integers exist for the rational tier ((n+3)/2 = 0.5n + 1.5);
# they show up in a quarter of constant draws, the rest are integers.
HALF_CONSTS = (-1.5, -0.5, 0.5, 1.5)
HALF_CONST_P = 0.25
EXP_RANGE = (-3, 3)   # constant exponent for the n^k form
BASE_RANGE = (-3, 3)  # constant base for the k^n form
MAX_INIT_DEPTH = 6


def _rand_const(rng: random.Random) -> Const:
    if rng.random() < HALF_CONST_P:
        return Const(rng.choice(HALF_CONSTS))
    return Const(float(rng.randint(*CONST_RANGE)))


def _rand_exp(rng: random.Random) -> Const:
    return Const(float(rng.randint(*EXP_RANGE)))


def _rand_base(rng: random.Random) -> Const:
    return Const(float(rng.randint(*BASE_RANGE)))


def random_tree(rng: random.Random, max_depth: int, method: str = "grow",
                ops: tuple[str, ...] = BINARY_OPS) -> Node:
    """Random tree of depth at most max_depth; method 'full' builds op
    nodes to the limit, 'grow' may terminate a branch early (the
    classic half-and-half pair). max_depth counts nodes, so 1 = leaf.
    ops limits which binary operators may appear."""
    if max_depth <= 1 or (method == "grow" and rng.random() < 0.25):
        return VAR if rng.random() < 0.5 else _rand_const(rng)
    op = rng.choice(ops)
    if op == "^":
        if rng.random() < 0.5:  # n^k form: constant exponent
            left = random_tree(rng, max_depth - 1, method, ops)
            right = _rand_exp(rng)
        else:  # k^n form: constant base
            left = _rand_base(rng)
            right = random_tree(rng, max_depth - 1, method, ops)
        return BinOp(op, left, right)
    left = random_tree(rng, max_depth - 1, method, ops)
    right = random_tree(rng, max_depth - 1, method, ops)
    return BinOp(op, left, right)


def subtree_at(node: Node, path: tuple[int, ...]) -> Node:
    for step in path:
        if isinstance(node, BinOp):
            node = node.left if step == 0 else node.right
        elif isinstance(node, Neg):
            node = node.child
        else:  # terminal: path overshoots, the terminal itself is the spot
            break
    return node


def replace_subtree(node: Node, path: tuple[int, ...], new: Node) -> Node:
    """Functional splice: returns a new tree, never mutates the input."""
    if not path:
        return new
    if isinstance(node, BinOp):
        if path[0] == 0:
            return BinOp(node.op, replace_subtree(node.left, path[1:], new), node.right)
        return BinOp(node.op, node.left, replace_subtree(node.right, path[1:], new))
    if isinstance(node, Neg) and path[0] == 0:
        return Neg(replace_subtree(node.child, path[1:], new))
    return node


def _all_paths(node: Node, path: tuple[int, ...] = ()):
    yield path
    if isinstance(node, BinOp):
        yield from _all_paths(node.left, path + (0,))
        yield from _all_paths(node.right, path + (1,))
    elif isinstance(node, Neg):
        yield from _all_paths(node.child, path + (0,))


def random_path(rng: random.Random, node: Node) -> tuple[int, ...]:
    """Path to a node chosen uniformly among ALL nodes — roots and
    internal nodes included, not just terminals. A descent that walks
    until it hits a leaf only ever mutates leaves, which silently
    cripples every genetic operator."""
    return rng.choice(list(_all_paths(node)))


def ensure_grammar(node: Node, rng: random.Random,
                   ops: tuple[str, ...] = BINARY_OPS) -> Node:
    """Post-crossover repair: ^ must keep a constant on one side. A splice
    can leave n*n under a ^; re-roll a constant exponent rather than
    rejecting the child (keeps operator flow simple). When ^ is disabled
    entirely, a bare ^ swaps to an allowed operator."""
    if isinstance(node, BinOp):
        left = ensure_grammar(node.left, rng, ops)
        right = ensure_grammar(node.right, rng, ops)
        if node.op == "^" and not isinstance(left, Const) \
                and not isinstance(right, Const):
            if "^" in ops:
                right = _rand_exp(rng)
            else:
                return BinOp(rng.choice(ops), left, right)
        return BinOp(node.op, left, right)
    if isinstance(node, Neg):
        return Neg(ensure_grammar(node.child, rng, ops))
    return node


def initial_population(rng: random.Random, pop_size: int,
                       max_depth: int = MAX_INIT_DEPTH,
                       ops: tuple[str, ...] = BINARY_OPS) -> list[Node]:
    """Ramped half-and-half: depths cycle 2..max_depth, alternating
    full/grow, so generation 0 mixes shallow and deep trees."""
    population = []
    depths = list(range(2, max_depth + 1))
    for i in range(pop_size):
        method = "full" if i % 2 == 0 else "grow"
        population.append(
            random_tree(rng, depths[i % len(depths)], method, ops))
    return population


def subtree_mutation(rng: random.Random, tree: Node,
                     max_depth: int = MAX_INIT_DEPTH,
                     ops: tuple[str, ...] = BINARY_OPS) -> Node:
    path = random_path(rng, tree)
    budget = max(1, max_depth - len(path))
    replacement = random_tree(rng, min(3, budget), "grow", ops)
    return ensure_grammar(replace_subtree(tree, path, replacement), rng, ops)


def shrink_mutation(rng: random.Random, tree: Node) -> Node:
    """Replace a random subtree with a terminal: an active shortening
    move for the post-solution shrink phase, where dropping junk like
    x/x subterms has to happen fast, not wait for lucky crossovers."""
    path = random_path(rng, tree)
    replacement = VAR if rng.random() < 0.7 else _rand_const(rng)
    return replace_subtree(tree, path, replacement)


def point_mutation(rng: random.Random, tree: Node,
                   ops: tuple[str, ...] = BINARY_OPS) -> Node:
    """Swap one node in place: a constant value, or an operator (with the
    ^ constant rule re-checked)."""
    path = random_path(rng, tree)
    target = subtree_at(tree, path)
    if isinstance(target, Const):
        new = _rand_const(rng)
    elif isinstance(target, BinOp):
        op = rng.choice(ops)
        left, right = target.left, target.right
        if (op == "^" and not isinstance(left, Const)
                and not isinstance(right, Const)):
            right = _rand_exp(rng)
        new = BinOp(op, left, right)
    elif isinstance(target, Neg):
        new = target.child  # drop the minus: a small simplifying mutation
    else:  # Var
        new = _rand_const(rng)
    return replace_subtree(tree, path, new)


def crossover(rng: random.Random, a: Node, b: Node,
              max_depth: int = MAX_INIT_DEPTH,
              ops: tuple[str, ...] = BINARY_OPS) -> Node:
    """Subtree swap; retries a few times to respect the depth limit,
    falls back to a copy of parent a (elitism still guards progress)."""
    for _ in range(6):
        child = replace_subtree(a, random_path(rng, a), subtree_at(b, random_path(rng, b)))
        child = ensure_grammar(child, rng, ops)
        if depth(child) <= max_depth:
            return child
    return a
