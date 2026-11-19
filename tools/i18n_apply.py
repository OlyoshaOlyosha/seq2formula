"""One-shot i18n builder (dev tool, not shipped).

Uses tools/ru_en_map.py (ru -> en) to:
1. rewrite the Cyrillic text nodes / title / placeholder / aria-label
   in both HTML files to English (the canonical UI language);
2. rewrite Cyrillic string literals in app.js / neuro.js to English;
3. emit static/_locales/ru/messages.json as the reversed map {en: ru}.

Idempotent in practice: after the rewrite there is no Cyrillic left to
match, so re-running is a no-op.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ru_en_map import MAP  # noqa: E402

ROOT = Path(r"C:\Dev\GitHub\seq2formula")

# sort longest-first so longer Russian strings win over their substrings
RULES = sorted(MAP.items(), key=lambda kv: -len(kv[0]))


def sub_html_attrs(s: str) -> str:
    pat = re.compile(
        r'(title|placeholder|aria-label|data-tip)="([^"]*[А-Яа-яЁё][^"]*)"')

    def rep(m: re.Match) -> str:
        val = m.group(2)
        en = MAP.get(val.strip())
        if en is None:
            return m.group(0)
        return f'{m.group(1)}="{en}"'
    return pat.sub(rep, s)


def sub_html_text(s: str) -> str:
    pat = re.compile(r"(>)([^<>]*[А-Яа-яЁё][^<>]*)(<)")

    def rep(m: re.Match) -> str:
        gt, text, lt = m.group(1), m.group(2), m.group(3)
        stripped = text.strip()
        en = MAP.get(stripped)
        if en is None:
            return m.group(0)
        lead = text[:len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()):]
        return f"{gt}{lead}{en}{trail}{lt}"
    return pat.sub(rep, s)


def sub_js(s: str) -> str:
    """JS: replace quoted Russian literals with English; leave
    template-literal pieces — they are handled by hand where needed."""
    for ru, en in RULES:
        for quoted in (f'"{ru}"', f"'{ru}'"):
            if quoted in s:
                s = s.replace(quoted, json.dumps(en, ensure_ascii=False))
    return s


def main():
    stats = {}
    for rel in ("static/index.html", "static/neuro.html"):
        path = ROOT / rel
        s = path.read_text(encoding="utf-8")
        before = s
        s = sub_html_text(sub_html_attrs(s))
        path.write_text(s, encoding="utf-8", newline="\n")
        stats[rel] = "changed" if s != before else "no-op"

    for rel in ("static/app.js", "static/neuro.js"):
        path = ROOT / rel
        s = path.read_text(encoding="utf-8")
        before = s
        s = sub_js(s)
        path.write_text(s, encoding="utf-8", newline="\n")
        stats[rel] = "changed" if s != before else "no-op"

    # static/_locales/ru/messages.json: reversed map, en key -> ru value
    loc = {en: ru for ru, en in MAP.items() if en != ru}
    loc_dir = ROOT / "static" / "_locales" / "ru"
    loc_dir.mkdir(parents=True, exist_ok=True)
    (loc_dir / "messages.json").write_text(
        json.dumps(loc, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8", newline="\n")
    

    print(json.dumps(stats, indent=1))
    print("locale entries:", len(loc))


if __name__ == "__main__":
    main()
