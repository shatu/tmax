#!/usr/bin/env python3
"""Prove the pass-side correction is purely subtractive against the real corpus.

"Strictly narrower by construction" is an argument; this is a measurement. If
the corrected token matches even ONE line the old one did not, the change can
invent findings and the claim in PROVENANCE is false.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool_audit import ANSI, _PASSED_TOKEN, load

OLD = re.compile(r"\b\d+ passed\b|=+ .*passed.* =+", re.I)
d, tag, mx = sys.argv[1], sys.argv[2], int(sys.argv[3])
old_hits = new_hits = only_new = only_old = 0
examples = []
for r in load(d):
    if int(r.get("step", -1)) + 1 > mx:
        continue
    ri = r.get("request_info") or {}
    blob = (ANSI.sub("", str(ri.get("tool_outputs") or "")) + "\n"
            + ANSI.sub("", str(ri.get("tool_errors") or "")))
    for line in blob.splitlines():
        o, n = bool(OLD.search(line)), bool(_PASSED_TOKEN.search(line))
        old_hits += o; new_hits += n
        if n and not o:
            only_new += 1
            if len(examples) < 5: examples.append(line.strip()[:140])
        elif o and not n:
            only_old += 1
print(f"=== {tag}  old={old_hits}  new={new_hits}  dropped={only_old}  ADDED={only_new}")
for e in examples: print(f"    ADDED: {e!r}")
print("  SUBTRACTIVE_OK" if only_new == 0 else "  NOT SUBTRACTIVE — change can invent findings")
