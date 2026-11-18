"""Random formula generation and genetic operators over the AST.

Grammar v1: terminals are n and numeric constants (integers plus a few
halves for the rational puzzles); binary ops are + - * / ^ mod. Power
constrains the EXPONENT: a constant exponent gives n^2-style polynomials
(and k^const forms for k^n puzzles), an exponent of exactly n gives the
(expr)^n family — Pell/Lucas bases — and a bare Var base there is n^n.
Still forbidden: an arbitrary expression as the exponent (n^(n+1),
2^99999). Every operator enforces the depth limit and repairs ^ nodes
whose exponent was damaged through a crossover splice.

Grammar v2 additions: unary functions (UnOp nodes — sqrt, cbrt, fact,
abs, sin, cos, tan, ln) and the special constants π / e. Functions draw
with a much lower weight than binary ops, so enabling them widens what
the search CAN express without drowning the classic puzzles. Disabled
functions are repaired away (stripped to their argument), disabled
specials simply never draw.

Every grammar function takes the same tail of keyword arguments:
ops (binary), funcs (unary), specials (π/e as float values). Empty
tuples mean "feature not enabled" and reproduce v1 behavior exactly.

Grammar v3 additions: comparison ops (< >) and the piecewise If node.
Comparisons join the ops tuple (they evaluate to 1.0 / 0.0); an If node
is drawn with a low probability whenever any comparison is enabled —
it is opt-in via the same protocol as floor, never a default.
"""

from __future__ import annotations

import random

from .nodes import (
    CMP_OPS,
    VAR,
    BinOp,
    Const,
    If,
    Neg,
    Node,
    UnOp,
    Var,
    depth,
    size,
)

BINARY_OPS = ("+", "-", "*", "/", "^", "mod")
# Small integers cover the classic puzzles; the LARGE_CONSTS bucket makes
# bigger coefficients reachable (1000*n for a "1000 2000 3000" sequence)
# without flooding normal formulas. Draw probabilities below set the
# balance: 65% small ints, 25% halves, 10% large.
CONST_RANGE = (-10, 10)
HALF_CONSTS = (-1.5, -0.5, 0.5, 1.5)
HALF_CONST_P = 0.25
LARGE_CONSTS = (20, 25, 50, 100, 1000)
LARGE_CONST_P = 0.10
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
# Chance that an internal node is a piecewise If fork when comparisons
# are enabled. Kept very low: an if triples the tree it touches, and
# the classic puzzles must stay solvable at the same speed with the
# chip left on.
IF_NODE_P = 0.05


def _if_enabled(ops: tuple[str, ...]) -> bool:
    """Comparisons in the ops tuple are what makes If legal (the cond
    is always drawn as a comparison)."""
    return any(op in CMP_OPS for op in ops)


def _pow_legal(left: Node, right: Node) -> bool:
    """^ legality: a constant on either side (n^k / k^(expr)) or an
    exponent of exactly n ((expr)^n — Pell/Lucas — including the bare
    n^n). Still banned: an arbitrary base with an arbitrary non-n
    exponent, the (n+n)^(n+n) monster source."""
    return (isinstance(left, Const) or isinstance(right, Const)
            or isinstance(right, Var))


def _rand_const(rng: random.Random, specials: tuple[float, ...] = ()) -> Const:
    if specials and rng.random() < SPECIAL_CONST_P:
        return Const(rng.choice(specials))
    roll = rng.random()
    if roll < LARGE_CONST_P:
        # large constants are reachable but rare: a formula naturally
        # uses 1, 2, 0.5 far more often than 1000, so the search should
        # try them far more often too
        return Const(float(rng.choice(LARGE_CONSTS) * rng.choice((-1, 1))))
    if roll < LARGE_CONST_P + HALF_CONST_P:
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
    if _if_enabled(ops) and max_depth >= 3 and rng.random() < IF_NODE_P:
        # Piecewise fork: the condition is always a comparison of two
        # subexpressions (two levels down), the branches get the rest
        cmp_ops = [op for op in ops if op in CMP_OPS]
        cond = BinOp(rng.choice(cmp_ops),
                     random_tree(rng, max_depth - 2, method, ops, funcs,
                                 specials),
                     random_tree(rng, max_depth - 2, method, ops, funcs,
                                 specials))
        then = random_tree(rng, max_depth - 1, method, ops, funcs, specials)
        otherwise = random_tree(rng, max_depth - 1, method, ops, funcs,
                                specials)
        return If(cond, then, otherwise)
    op = rng.choice(ops)
    if op == "^":
        r = rng.random()
        if r < 0.45:  # n^k form: constant exponent
            left = random_tree(rng, max_depth - 1, method, ops, funcs, specials)
            right = _rand_exp(rng)
        elif r < 0.90:  # k^n form: constant base
            left = _rand_base(rng)
            right = random_tree(rng, max_depth - 1, method, ops, funcs, specials)
        else:  # (expr)^n form: exponent is n itself (Pell/Lucas family);
              # a bare Var base here is the n^n family, so it comes free
            left = random_tree(rng, max_depth - 1, method, ops, funcs, specials)
            right = VAR
        return BinOp(op, left, right)
    left = random_tree(rng, max_depth - 1, method, ops, funcs, specials)
    right = random_tree(rng, max_depth - 1, method, ops, funcs, specials)
    return BinOp(op, left, right)


def subtree_at(node: Node, path: tuple[int, ...]) -> Node:
    for step in path:
        if isinstance(node, BinOp):
            node = node.left if step == 0 else node.right
        elif isinstance(node, If):
            node = (node.cond, node.then, node.otherwise)[step]
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
    if isinstance(node, If):
        if path[0] == 0:
            return If(replace_subtree(node.cond, path[1:], new),
                      node.then, node.otherwise)
        if path[0] == 1:
            return If(node.cond, replace_subtree(node.then, path[1:], new),
                      node.otherwise)
        return If(node.cond, node.then,
                  replace_subtree(node.otherwise, path[1:], new))
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
    path: list[int] = []
    while k:
        k -= 1
        if isinstance(node, BinOp):
            left_size = size(node.left)
            if k < left_size:
                node = node.left
                path.append(0)
            else:
                k -= left_size
                node = node.right
                path.append(1)
        elif isinstance(node, If):
            cond_size = size(node.cond)
            then_size = size(node.then)
            if k < cond_size:
                node = node.cond
                path.append(0)
            elif k < cond_size + then_size:
                k -= cond_size
                node = node.then
                path.append(1)
            else:
                k -= cond_size + then_size
                node = node.otherwise
                path.append(2)
        elif isinstance(node, (Neg, UnOp)):
            node = node.child
            path.append(0)
        else:  # leaf: k was within the subtree, so it must be 0 here
            break
    return tuple(path)


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
        if node.op == "^":
            if "^" not in ops:
                return BinOp(rng.choice(ops), left, right)
            if not _pow_legal(left, right):
                right = _rand_exp(rng)
        return BinOp(node.op, left, right)
    if isinstance(node, If):
        cond = _repair_splice(node.cond, rest, rng, ops) if step == 0 \
            else node.cond
        then = _repair_splice(node.then, rest, rng, ops) if step == 1 \
            else node.then
        otherwise = _repair_splice(node.otherwise, rest, rng, ops) \
            if step == 2 else node.otherwise
        # invariant: a fork's condition is always a comparison — a splice
        # can graft a whole fork (or any expression) into the cond slot;
        # normalize with an explicit > 0 test (the evaluator's truthiness)
        if not (isinstance(cond, BinOp) and cond.op in CMP_OPS):
            cond = BinOp(">", cond, Const(0.0))
        return If(cond, then, otherwise)
    if isinstance(node, UnOp):
        return UnOp(node.name, _repair_splice(node.child, rest, rng, ops))
    if isinstance(node, Neg):
        return Neg(_repair_splice(node.child, rest, rng, ops))
    return node


def ensure_grammar(node: Node, rng: random.Random,
                   ops: tuple[str, ...] = BINARY_OPS,
                   funcs: tuple[str, ...] = (),
                   specials: tuple[float, ...] = ()) -> Node:
    """Post-crossover repair: ^ must keep its exponent a constant or
    exactly n, and a function node whose function got disabled (or an
    unknown one from a corrupt splice) strips to its argument. A splice
    can leave an arbitrary expression as a ^ exponent; re-roll a constant
    exponent rather than rejecting the child (keeps operator flow
    simple). When ^ is disabled entirely, a bare ^
    swaps to an allowed operator. Comparisons disabled: a comparison node
    strips to its left side and an If fork keeps just the then branch."""
    if isinstance(node, BinOp):
        left = ensure_grammar(node.left, rng, ops, funcs, specials)
        right = ensure_grammar(node.right, rng, ops, funcs, specials)
        if node.op == "^":
            if "^" not in ops:
                return BinOp(rng.choice(ops), left, right)
            if not _pow_legal(left, right):
                right = _rand_exp(rng)
        if node.op in CMP_OPS and node.op not in ops:
            return left  # strip the comparison, keep the left operand
        return BinOp(node.op, left, right)
    if isinstance(node, If):
        cond = ensure_grammar(node.cond, rng, ops, funcs, specials)
        then = ensure_grammar(node.then, rng, ops, funcs, specials)
        otherwise = ensure_grammar(node.otherwise, rng, ops, funcs, specials)
        if not _if_enabled(ops):
            return then  # the fork is illegal: keep one branch
        # invariant: a fork's condition is always a comparison — a splice
        # can graft a whole fork (or any expression) into the cond slot;
        # normalize with an explicit > 0 test (the evaluator's truthiness)
        if not (isinstance(cond, BinOp) and cond.op in CMP_OPS):
            cond = BinOp(">", cond, Const(0.0))
        return If(cond, then, otherwise)
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
        if op == "^" and not _pow_legal(left, right):
            right = _rand_exp(rng)
        new = BinOp(op, left, right)
    elif isinstance(target, If):
        # Swap the branches: the cheapest restructure of a fork (and the
        # one that flips which half of the domain gets which formula)
        new = If(target.cond, target.otherwise, target.then)
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


def root_mutation(rng: random.Random, tree: Node,
                  ops: tuple[str, ...] = BINARY_OPS,
                  funcs: tuple[str, ...] = (),
                  specials: tuple[float, ...] = ()) -> Node:
    """Structural jump: wrap the whole tree in a new root operation.

    Random trees are mostly invalid (div-zero / domain) and die before
    they can express anything new. Wrapping the current best in a new
    operation (sqrt(best), best + n, best mod 3) guarantees a valid
    formula and forces exploration of a new structural family."""
    if funcs and rng.random() < 0.5:
        name = rng.choice(funcs)
        return UnOp(name, tree)
    op = rng.choice(ops)
    if op == "^":
        r = rng.random()
        if r < 0.4:
            return BinOp(op, tree, _rand_exp(rng))
        if r < 0.8:
            return BinOp(op, _rand_base(rng), tree)
        return BinOp(op, tree, VAR)  # (expr)^n: Pell/Lucas family
    other = random_tree(rng, 2, "grow", ops, funcs, specials)
    if rng.random() < 0.5:
        return BinOp(op, tree, other)
    return BinOp(op, other, tree)
