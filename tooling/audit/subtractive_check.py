#!/usr/bin/env python3
"""Measure what the pass-side correction adds and drops against the real corpus.

"Strictly narrower by construction" is an argument; this is a measurement. An
earlier fix asserted subtractiveness and silently ADDED 198 lines by matching
the "1" in "test_..._defaults_to_1 PASSED"; this check caught it.

SCOPE, per hamishivi: the zero-count and digitless-banner rules must be purely
subtractive, and a violation there is a hard failure. Equal-fraction and
label-prefix acceptance are INTENTIONAL, calibrated additions under the 06:06
ruling, so lines added solely by those rules are reported and enumerated rather
than treated as accidental expansion. Blocking them would block the ruling.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool_audit import ANSI, _PASSED_TOKEN, load

OLD = re.compile(r"\b\d+ passed\b|=+ .*passed.* =+", re.I)
d, tag, mx = sys.argv[1], sys.argv[2], int(sys.argv[3])
old_hits = new_hits = only_new = only_old = 0
examples = []
examples_all = []
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
            examples_all.append(line.strip()[:140])
        elif o and not n:
            only_old += 1
FRAC = re.compile(r"(?<![\w/.\-])(\d+)/(\d+) passed\b", re.I)
LABEL = re.compile(r"[A-Za-z][\w .\-]*:\s*(?!0+\b)\d+ passed\b", re.I)


def intended(line):
    """Is this line added by a RULED addition rather than by accident?"""
    m = FRAC.search(line)
    if m and int(m.group(1)) == int(m.group(2)) and int(m.group(1)) > 0:
        return "equal_fraction"
    if LABEL.search(line):
        return "label_prefix"
    return None


intended_adds = {}
unexpected = []
for e in examples_all:
    k = intended(e)
    if k:
        intended_adds[k] = intended_adds.get(k, 0) + 1
    else:
        unexpected.append(e)

print(f"=== {tag}  old={old_hits}  new={new_hits}  dropped={only_old}  added={only_new}")
print(f"    added by RULED rules : {intended_adds or 'none'}")
print(f"    added UNEXPECTEDLY   : {len(unexpected)}")
for e in unexpected[:5]:
    print(f"      {e!r}")
print("  SCOPED_SUBTRACTIVE_OK" if not unexpected
      else "  UNEXPECTED EXPANSION — additions outside the ruled rules")
