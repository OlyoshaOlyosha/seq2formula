"""i18n coverage validator (dev tool): every English UI string in the
markup and JS literals must have a Russian translation in
static/_locales/ru/messages.json. Run after any UI edit:

    python tools/i18n_validate.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(r"C:\Dev\GitHub\seq2formula")

strings: set[str] = set()

for rel in ("static/index.html", "static/neuro.html"):
    s = (ROOT / rel).read_text(encoding="utf-8")
    for pat in (r'(?:title|placeholder|aria-label|data-tip)="([^"]+)"',
                r">([^<>]+)<"):
        for m in re.finditer(pat, s):
            v = m.group(1).strip()
            if re.search(r"[A-Za-z]", v) and not v.startswith(("<!--", "<")) \
                    and not re.search(r"[А-Яа-яЁё]", v):
                strings.add(v)

for rel in ("static/app.js", "static/neuro.js"):
    s = (ROOT / rel).read_text(encoding="utf-8")
    for m in re.finditer(r'"([^"\n]+)"', s):
        v = m.group(1)
        if re.search(r"[A-Za-z]{3}", v) and " " in v.strip() \
                and not re.search(r"[А-Яа-яЁё]", v) \
                and not re.match(r"^[a-z_:/.-]+$", v) \
                and "{" not in v and "http" not in v:
            strings.add(v.strip())

locale = json.loads(
    (ROOT / "static/_locales/ru/messages.json").read_text(encoding="utf-8"))

# code/CSS fragments captured by the regexes are not UI strings;
# " and " is a concatenation fragment whose full key lives in the locale
fragment = re.compile(r"[()<>{};=+]|^\.|\{\d|^[a-z-]+( [a-z-]+)+$")
def covered(s: str) -> bool:
    # the trailing space of a concatenation prefix is part of the locale
    # key but invisible to the trimmed regex capture
    return (s in locale or s + " " in locale or s.rstrip() in locale
            or s.strip() in locale)


missing = sorted(s for s in strings
                 if not covered(s) and not fragment.search(s)
                 and s not in ("RU", "cos", "ln", "sin", "tan", "if", "pi", "e",
                               "seq2formula",
                               "xʸ", "|x|", "⌊x⌋"))
print(f"UI strings: {len(strings)}, locale: {len(locale)}, missing: {len(missing)}")
for s in missing:
    print(" MISSING:", repr(s))
