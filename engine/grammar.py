"""Random formula generation and genetic operators over the AST.

Grammar v1: terminals are n and numeric constants (integers plus a few
halves for the rational puzzles); binary ops are + - * / ^ mod. Power
needs a constant on at least one side: a constant exponent gives n^2-style
polynomials, a constant base gives 2^n-style exponentials; n^n stays
forbidden. Every operator enforces the depth limit and repairs ^ nodes
that lost their constant through a crossover splice.

Grammar v2 additions: unary functions (UnOp nodes — sqrt, cbrt, fact,
abs, sin, cos, tan, ln) and the special constants π / e. Functions draw
with a much lower weight than binary ops, so enabling them widens what
the search CAN express without drowning the classic puzzles. Disabled
functions are repaired away (stripped to their argument), disabled
specials simply never draw.

Every grammar function takes the same tail of keyword arguments:
ops (binary), funcs (unary), specials (π/e as float values). Empty
tuples mean "feature not enabled" and reproduce v1 behavior exactly.
"""

from __future__ import annotations

import random

from .nodes import VAR, BinOp, Const, Neg, Node, UnOp, Var, depth, size

BINARY_OPS = ("+", "-", "*", "/", "^", "mod")
CONST_RANGE = (-5, 5)
# Half-integers exist for the rational tier ((n+3)/2 = 0.5n + 1.5);
# they show up in a quarter of constant draws, the rest are integers.
HALF_CONSTS = (-1.5, -0.5, 0.5, 1.5)
HALF_CONST_P = 0.25
EXP_RANGE = (-3, 3)   # constant exponent for the n^k form
BASE_RANGE = (-3, 3)  # constant base for the k^n form
MAX_INIT_DEPTH = 6
# Chance that an internal node is a function application when any
# functions are enabled. Kept low on purpose: functions multiply the
# search space, and the classic puzzles must stay solvable at the same
# speed when they are left enabled.
UNARY_NODE_P = 0.18
# Chance that a constant draw is π or e (when specials are enabled).
SPECIAL_CONST_P = 0.12


def _rand_const(rng: random.Random, specials: tuple[float, ...] = ()) -> Const:
    if specials and rng.random() < SPECIAL_CONST_P:
        return Const(rng.choice(specials))
    if rng.random() < HALF_CONST_P:
        return Const(rng.choice(HALF_CONSTS))
    return Const(float(rng.randint(*CONST_RANGE)))


def _rand_exp(rng: random.Random) -> Const:
    return Const(float(rng.randint(*EXP_RANGE)))


def _rand_base(rng: random.Random) -> Const:
    return Const(float(rng.randint(*BASE_RANGE)))


def _wrap_func(name: str, child: Node) -> Node:
    """fact of a random subtree evaluates to a non-integer or an
    overflow almost always — as an init draw it is pure junk. Keep it
    only on a bare variable or a small non-negative integer constant
    (n! and k! are exactly the useful forms); anywhere else drop the
    wrapper and hand the child back."""
    if name != "fact":
        return UnOp(name, child)
    if isinstance(child, Var):
        return UnOp(name, child)
    if isinstance(child, Const) and 0 <= child.value <= 170 \
            and child.value == int(child.value):
        return UnOp(name, child)
    return child


def random_tree(rng: random.Random, max_depth: int, method: str = "grow",
                ops: tuple[str, ...] = BINARY_OPS,
                funcs: tuple[str, ...] = (),
                specials: tuple[float, ...] = ()) -> Node:
    """Random tree of depth at most max_depth; method 'full' builds op
    nodes to the limit, 'grow' may terminate a branch early (the
    classic half-and-half pair). max_depth counts nodes, so 1 = leaf.
    ops limits binary operators, funcs unary functions, specials the
    π/e terminal pool."""
    if max_depth <= 1 or (method == "grow" and rng.random() < 0.25):
        return VAR if rng.random() < 0.5 else _rand_const(rng, specials)
    if funcs and rng.random() < UNARY_NODE_P:
        name = rng.choice(funcs)
        child = random_tree(rng, max_depth - 1, method, ops, funcs, specials)
        return _wrap_func(name, child)
    op = rng.choice(ops)
    if op == "^":
        if rng.random() < 0.5:  # n^k form: constant exponent
            left = random_tree(rng, max_depth - 1, method, ops, funcs, specials)
            right = _rand_exp(rng)
        else:  # k^n form: constant base
            left = _rand_base(rng)
            right = random_tree(rng, max_depth - 1, method, ops, funcs, specials)
        return BinOp(op, left, right)
    left = random_tree(rng, max_depth - 1, method, ops, funcs, specials)
    right = random_tree(rng, max_depth - 1, method, ops, funcs, specials)
    return BinOp(op, left, right)


def subtree_at(node: Node, path: tuple[int, ...]) -> Node:
    for step in path:
        if isinstance(node, BinOp):
            node = node.left if step == 0 else node.right
        elif isinstance(node, (Neg, UnOp)):
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
    if isinstance(node, UnOp) and path[0] == 0:
        return UnOp(node.name, replace_subtree(node.child, path[1:], new))
    if isinstance(node, Neg) and path[0] == 0:
        return Neg(replace_subtree(node.child, path[1:], new))
    return node


def random_path(rng: random.Random, node: Node) -> tuple[int, ...]:
    """Path to a node chosen uniformly among ALL nodes — roots and
    internal nodes included, not just terminals. A descent that walks
    until it hits a leaf only ever mutates leaves, which silently
    cripples every genetic operator.

    Picks the k-th node in preorder (the same order the old path-listing
    walk produced) with a single counting descent — building the whole
    path list per operator call was a hot-loop cost."""
    k = rng.randrange(size(node))
    path: tuple[int, ...] = ()
    while k:
        k -= 1
        if isinstance(node, BinOp):
            left_size = size(node.left)
            if k < left_size:
                node, path = node.left, path + (0,)
            else:
                k -= left_size
                node, path = node.right, path + (1,)
        elif isinstance(node, (Neg, UnOp)):
            node, path = node.child, path + (0,)
        else:  # leaf: k was within the subtree, so it must be 0 here
            break
    return path


def _repair_splice(node: Node, path: tuple[int, ...], rng: random.Random,
                   ops: tuple[str, ...]) -> Node:
    """Repair a freshly spliced tree by walking only the splice path.

    Parents are always valid grammar (the population invariant), so the
    only damage a splice can do is under a `^` that sits on the path and
    lost its constant child to the graft. Re-roll a constant exponent
    there — or swap the operator when ^ is disabled. The grafted subtree
    itself was built by valid operators and needs no walk."""
    if not path:
        return node  # the graft point: replacement is valid by construction
    step, rest = path[0], path[1:]
    if isinstance(node, BinOp):
        if step == 0:
            left = _repair_splice(node.left, rest, rng, ops)
            right = node.right
        else:
            left = node.left
            right = _repair_splice(node.right, rest, rng, ops)
        if node.op == "^" and not isinstance(left, Const) \
                and not isinstance(right, Const):
            if "^" in ops:
                right = _rand_exp(rng)
            else:
                return BinOp(rng.choice(ops), left, right)
        return BinOp(node.op, left, right)
    if isinstance(node, UnOp):
        return UnOp(node.name, _repair_splice(node.child, rest, rng, ops))
    if isinstance(node, Neg):
        return Neg(_repair_splice(node.child, rest, rng, ops))
    return node


def ensure_grammar(node: Node, rng: random.Random,
                   ops: tuple[str, ...] = BINARY_OPS,
                   funcs: tuple[str, ...] = (),
                   specials: tuple[float, ...] = ()) -> Node:
    """Post-crossover repair: ^ must keep a constant on one side, and a
    function node whose function got disabled (or an unknown one from a
    corrupt splice) strips to its argument. A splice can leave n*n under
    a ^; re-roll a constant exponent rather than rejecting the child
    (keeps operator flow simple). When ^ is disabled entirely, a bare ^
    swaps to an allowed operator."""
    if isinstance(node, BinOp):
        left = ensure_grammar(node.left, rng, ops, funcs, specials)
        right = ensure_grammar(node.right, rng, ops, funcs, specials)
        if node.op == "^" and not isinstance(left, Const) \
                and not isinstance(right, Const):
            if "^" in ops:
                right = _rand_exp(rng)
            else:
                return BinOp(rng.choice(ops), left, right)
        return BinOp(node.op, left, right)
    if isinstance(node, UnOp):
        child = ensure_grammar(node.child, rng, ops, funcs, specials)
        if node.name not in funcs:
            return child  # strip: the argument is always valid grammar
        return UnOp(node.name, child)
    if isinstance(node, Neg):
        return Neg(ensure_grammar(node.child, rng, ops, funcs, specials))
    return node


def initial_population(rng: random.Random, pop_size: int,
                       max_depth: int = MAX_INIT_DEPTH,
                       ops: tuple[str, ...] = BINARY_OPS,
                       funcs: tuple[str, ...] = (),
                       specials: tuple[float, ...] = ()) -> list[Node]:
    """Ramped half-and-half: depths cycle 2..max_depth, alternating
    full/grow, so generation 0 mixes shallow and deep trees."""
    population = []
    depths = list(range(2, max_depth + 1))
    for i in range(pop_size):
        method = "full" if i % 2 == 0 else "grow"
        population.append(
            random_tree(rng, depths[i % len(depths)], method, ops,
                        funcs, specials))
    return population


def subtree_mutation(rng: random.Random, tree: Node,
                     max_depth: int = MAX_INIT_DEPTH,
                     ops: tuple[str, ...] = BINARY_OPS,
                     funcs: tuple[str, ...] = (),
                     specials: tuple[float, ...] = ()) -> Node:
    path = random_path(rng, tree)
    budget = max(1, max_depth - len(path))
    replacement = random_tree(rng, min(3, budget), "grow", ops, funcs, specials)
    return _repair_splice(replace_subtree(tree, path, replacement), path,
                          rng, ops)


def shrink_mutation(rng: random.Random, tree: Node,
                    specials: tuple[float, ...] = ()) -> Node:
    """Replace a random subtree with a terminal: an active shortening
    move for the post-solution shrink phase, where dropping junk like
    x/x subterms has to happen fast, not wait for lucky crossovers."""
    path = random_path(rng, tree)
    replacement = VAR if rng.random() < 0.7 else _rand_const(rng, specials)
    return _repair_splice(replace_subtree(tree, path, replacement), path,
                          rng, BINARY_OPS)


def point_mutation(rng: random.Random, tree: Node,
                   ops: tuple[str, ...] = BINARY_OPS,
                   funcs: tuple[str, ...] = (),
                   specials: tuple[float, ...] = ()) -> Node:
    """Swap one node in place: a constant value, an operator (with the
    ^ constant rule re-checked), or a function choice."""
    path = random_path(rng, tree)
    target = subtree_at(tree, path)
    if isinstance(target, Const):
        new = _rand_const(rng, specials)
    elif isinstance(target, BinOp):
        op = rng.choice(ops)
        left, right = target.left, target.right
        if (op == "^" and not isinstance(left, Const)
                and not isinstance(right, Const)):
            right = _rand_exp(rng)
        new = BinOp(op, left, right)
    elif isinstance(target, UnOp):
        if funcs:
            new = UnOp(rng.choice(funcs), target.child)
        else:
            new = target.child  # drop the call: a small simplifying move
    elif isinstance(target, Neg):
        new = target.child  # drop the minus: a small simplifying mutation
    else:  # Var
        new = _rand_const(rng, specials)
    return replace_subtree(tree, path, new)


def crossover(rng: random.Random, a: Node, b: Node,
              max_depth: int = MAX_INIT_DEPTH,
              ops: tuple[str, ...] = BINARY_OPS,
              funcs: tuple[str, ...] = (),
              specials: tuple[float, ...] = ()) -> Node:
    """Subtree swap; retries a few times to respect the depth limit,
    falls back to a copy of parent a (elitism still guards progress)."""
    for _ in range(6):
        path = random_path(rng, a)
        child = replace_subtree(a, path, subtree_at(b, random_path(rng, b)))
        child = _repair_splice(child, path, rng, ops)
        if depth(child) <= max_depth:
            return child
    return a
