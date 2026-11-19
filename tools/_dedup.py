"""Dedupe repeated keys in tools/ru_en_map.py (first occurrence wins)."""
import json
import re
from pathlib import Path

p = Path(__file__).parent / "ru_en_map.py"
lines = p.read_text(encoding="utf-8").split("\n")
seen = set()
out = []
removed = 0
key_re = re.compile(r'^    ("(?:[^"\\]|\\.)*"): ')
for line in lines:
    m = key_re.match(line)
    if m:
        key = json.loads(m.group(1))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
    out.append(line)
p.write_text("\n".join(out), encoding="utf-8", newline="\n")
print("removed dupes:", removed)
