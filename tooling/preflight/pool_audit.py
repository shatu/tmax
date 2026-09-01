#!/usr/bin/env python3
"""Full SIF-pool coverage audit — the pre-launch gate.

hamishivi's stated bar (tmax-private#1): 14,488/14,488 resolver-based matches,
non-empty, `apptainer inspect` valid, zero bare/docker refs — plus the size
floor I added after a `du`-reported "2.0K" turned out to be a 131 MB file whose
blocks had not yet flushed. That scare exposed the real hole: "non-empty +
inspect passes" would accept a TRUNCATED SIF, so a floor calibrated on the
observed distribution is part of the gate rather than a nicety.

Resolution is driven through the tree's own `prefer_local_sif()` so the audit
cannot drift from what the trainer will actually do at runtime. The fallback
escape hatch is force-disabled here: if the guard is present a miss raises, and
if it is absent a miss returns a bare ref — both are recorded as failures, so
this audit gives the same verdict on a guarded or unguarded tree.

`apptainer inspect` is NOT sampled. It runs over every image in a parallel
array (probes/inspect_all.sh) and this script consumes those shard reports, so
the inspect gate covers the same 14,488 as the coverage gate.

Usage:
  python3 pool_audit.py --tree <open-instruct> --manifest <tsv> --pool <dir> \
      [--inspect-reports 'logs/inspect-*.jsonl'] [--json out.json]
Exit 0 only if every gate passes.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import logging
import os
import re
import statistics
import sys

BARE_RE = re.compile(r"^(docker://|oras://|[^/]+/[^:]+:)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--inspect-reports", default=None,
                    help="glob of jsonl shard reports from inspect_all.sh")
    ap.add_argument("--floor-mb", type=float, default=10.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.tree)
    os.environ["SWERL_APPTAINER_SIF_DIR"] = args.pool
    # never let the escape hatch mask a miss during an audit
    os.environ.pop("SWERL_ALLOW_REMOTE_IMAGE_FALLBACK", None)
    logging.disable(logging.WARNING)
    from open_instruct.environments.apptainer_images import prefer_local_sif

    images = [l.split("\t")[0].strip() for l in open(args.manifest) if l.strip()]
    resolved, missing, bare = {}, [], []
    for im in images:
        try:
            r = prefer_local_sif(im)
        except Exception:
            missing.append(im)          # guard raised: a genuine miss
            continue
        if r.endswith(".sif") and os.path.isfile(r):
            resolved[im] = r
        else:
            bare.append(im)             # unguarded tree returned a remote ref

    sizes = {im: os.path.getsize(p) for im, p in resolved.items()}
    nonzero = [s for s in sizes.values() if s > 0]
    empty = sorted(im for im, s in sizes.items() if s == 0)

    floor = int(args.floor_mb * 1024 * 1024)
    p50 = statistics.median(nonzero) if nonzero else 0
    undersized = sorted(((s, im) for im, s in sizes.items() if 0 < s < floor))

    # any resolved path that is still a registry reference rather than a file
    bare_like = sorted(im for im, p in resolved.items() if BARE_RE.match(p))

    # full-pool apptainer inspect, consumed from the array shards
    ins = {"checked": 0, "failed": [], "shards": 0}
    if args.inspect_reports:
        for f in sorted(globmod.glob(args.inspect_reports)):
            ins["shards"] += 1
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ins["checked"] += 1
                if rec.get("rc") != 0:
                    ins["failed"].append(rec)
    inspect_complete = ins["checked"] >= len(resolved) > 0

    gates = {
        "coverage_all_resolved": len(missing) == 0 and len(bare) == 0
                                 and len(resolved) == len(images),
        "zero_bare_or_docker_refs": len(bare) == 0 and len(bare_like) == 0,
        "zero_empty": len(empty) == 0,
        f"size_floor_{args.floor_mb:g}MB": len(undersized) == 0,
        "inspect_full_coverage": inspect_complete,
        "inspect_all_valid": len(ins["failed"]) == 0 and ins["checked"] > 0,
    }
    ok = all(gates.values())

    report = {
        "tree": args.tree,
        "pool": args.pool,
        "manifest_images": len(images),
        "resolved": len(resolved),
        "missing": missing[:20],
        "missing_count": len(missing),
        "bare_refs": bare[:20],
        "bare_ref_count": len(bare) + len(bare_like),
        "empty": empty[:20],
        "empty_count": len(empty),
        "undersized": [{"image": im, "bytes": s} for s, im in undersized[:20]],
        "undersized_count": len(undersized),
        "size_min": min(nonzero) if nonzero else None,
        "size_p50": p50,
        "size_max": max(nonzero) if nonzero else None,
        "total_bytes": sum(nonzero),
        "inspect_checked": ins["checked"],
        "inspect_shards": ins["shards"],
        "inspect_failures": ins["failed"][:20],
        "inspect_failure_count": len(ins["failed"]),
        "gates": gates,
        "VERDICT": "PASS" if ok else "FAIL",
    }
    if args.json:
        json.dump(report, open(args.json, "w"), indent=1)

    print(f"tree                : {args.tree}")
    print(f"manifest images     : {len(images)}")
    print(f"resolved via tree   : {len(resolved)}")
    print(f"missing             : {len(missing)}")
    print(f"bare/docker refs    : {len(bare) + len(bare_like)}")
    print(f"empty files         : {len(empty)}")
    print(f"under {args.floor_mb:g} MB floor  : {len(undersized)}")
    if nonzero:
        print(f"sizes min/p50/max   : {min(nonzero)/1e6:.1f} / {p50/1e6:.1f} / "
              f"{max(nonzero)/1e6:.1f} MB")
    print(f"total pool bytes    : {sum(nonzero)/1e12:.2f} TB")
    print(f"apptainer inspect   : {ins['checked']} checked over {ins['shards']} shards, "
          f"{len(ins['failed'])} failed")
    print()
    for g, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {g}")
    print(f"\nVERDICT: {report['VERDICT']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
