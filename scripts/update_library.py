"""Verify and expand sequences.json using explicit mathematical formulas."""

import json
import math
import re
from pathlib import Path

# Add the project root to sys.path so we can find `sequences.json`
ROOT = Path(__file__).resolve().parent.parent


def parse_ref(text: str) -> tuple:
    """A tiny parser to convert the human-readable reference formula to an AST tuple."""
    text = text.replace("π", " pi ").replace("·", " * ").replace("mod", " % ")
    tokens = re.findall(r'[a-zA-Z_]+|\d+\.\d+|\d+|[-+*/^%()!]', text)
    pos = 0
    
    def peek():
        return tokens[pos] if pos < len(tokens) else None
        
    def consume(expected=None):
        nonlocal pos
        tok = peek()
        if expected and tok != expected:
            raise ValueError(f"Expected {expected}, got {tok}")
        pos += 1
        return tok
    
    def parse_expr():
        left = parse_term()
        while peek() in ('+', '-'):
            op = consume()
            right = parse_term()
            left = (op, left, right)
        return left
        
    def parse_term():
        left = parse_factor()
        while True:
            p = peek()
            if p in ('*', '/', '%'):
                op = consume()
                right = parse_factor()
                left = (op, left, right)
            elif p is not None and (p[0].isalnum() or p in ('(', 'pi', 'e')):
                # implicit multiplication (e.g. "2n" or "n(n+1)")
                right = parse_factor()
                left = ('*', left, right)
            else:
                break
        return left
        
    def parse_factor():
        left = parse_primary()
        if peek() == '^':
            consume('^')
            right = parse_factor()  # right associative
            return ('^', left, right)
        return left
        
    def parse_primary():
        tok = peek()
        if tok == '(':
            consume('(')
            node = parse_expr()
            consume(')')
        elif tok == '-':
            consume('-')
            node = ('neg', parse_primary())
        elif tok in ('sqrt', 'cbrt', 'ln', 'abs', 'sin', 'cos'):
            func = consume()
            consume('(')
            arg = parse_expr()
            consume(')')
            node = ('func', func, arg)
        elif tok == 'n':
            consume()
            node = ('n',)
        elif tok == 'pi':
            consume()
            node = ('const', math.pi)
        elif tok == 'e':
            consume()
            node = ('const', math.e)
        elif tok and tok[0].isdigit():
            consume()
            node = ('const', float(tok))
        else:
            raise ValueError(f"Unexpected token: {tok}")
            
        while peek() == '!':
            consume('!')
            node = ('fact', node)
            
        return node
        
    return parse_expr()


def evaluate_ast(node: tuple, n: float) -> float:
    """Recursively evaluate the AST tuple for a given n."""
    t = node[0]
    if t == 'n':
        return n
    if t == 'const':
        return node[1]
    if t == 'neg':
        return -evaluate_ast(node[1], n)
    if t == 'func':
        val = evaluate_ast(node[2], n)
        if node[1] == 'sqrt':
            return math.sqrt(val)
        if node[1] == 'cbrt':
            return math.copysign(abs(val)**(1/3), val)
        if node[1] == 'ln':
            return math.log(val)
        if node[1] == 'abs':
            return abs(val)
        if node[1] == 'sin':
            return math.sin(val)
        if node[1] == 'cos':
            return math.cos(val)
    if t == 'fact':
        return math.factorial(int(round(evaluate_ast(node[1], n))))
    if t == '+':
        return evaluate_ast(node[1], n) + evaluate_ast(node[2], n)
    if t == '-':
        return evaluate_ast(node[1], n) - evaluate_ast(node[2], n)
    if t == '*':
        return evaluate_ast(node[1], n) * evaluate_ast(node[2], n)
    if t == '/':
        return evaluate_ast(node[1], n) / evaluate_ast(node[2], n)
    if t == '%':
        return evaluate_ast(node[1], n) % evaluate_ast(node[2], n)
    if t == '^':
        return evaluate_ast(node[1], n) ** evaluate_ast(node[2], n)
    raise ValueError(f"Unknown node type: {t}")


def verify_and_expand():
    print("Loading sequences.json...")
    with open(ROOT / "sequences.json", encoding="utf-8") as f:
        data = json.load(f)

    errors = 0
    modified = False
    
    for p in data["puzzles"]:
        pid = p["id"]
        ref_str = p.get("reference")
        if not ref_str:
            print(f"⚠️  No reference formula for {pid}")
            continue
        
        try:
            ast = parse_ref(ref_str)
        except Exception as e:
            print(f"❌ Failed to parse reference for {pid}: {e}")
            continue

        new_terms = []
        
        for i in range(1, 11):
            try:
                val = evaluate_ast(ast, i)
                if abs(val) > 1e9:
                    break
                if isinstance(val, float):
                    if val.is_integer():
                        val = int(val)
                    else:
                        val = round(val, 6)
                new_terms.append(val)
            except Exception as e:
                print(f"❌ Error evaluating {pid} at n={i}: {e}")
                break
        
        old_terms = p["terms"]
        for i, (t, e) in enumerate(zip(old_terms, new_terms, strict=False)):
            if abs(t - e) > 1e-5:
                print(f"❌ Mismatch in {pid} at n={i+1}: expected {e}, got {t}")
                errors += 1
                break
        
        if old_terms != new_terms:
            p["terms"] = new_terms
            print(f"✅ {pid} updated to {len(new_terms)} terms")
            modified = True
        else:
            print(f"⏭️  {pid} unchanged ({len(new_terms)} terms)")

    if modified:
        with open(ROOT / "sequences.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("\nDone! sequences.json updated.")
    else:
        print("\nDone! All sequences are already up to date (file untouched).")
        
    if errors > 0:
        print(f"Found {errors} mathematical errors in the old data (they are now fixed).")
    else:
        print("All original formulas were mathematically correct.")


if __name__ == "__main__":
    verify_and_expand()