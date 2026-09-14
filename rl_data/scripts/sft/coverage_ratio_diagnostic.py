"""Coverage-ratio diagnostic (rho_K) from TailSFT (arXiv:2608.25756, Def 4.1).

Compares a base model against a standard-SFT checkpoint on the same task
subset, using the per-task rollout summaries produced by
``rl_data.generate_solutions`` (``task_*/solutions/<tag>*_summary.json``).

For each task i with c successes out of n rollouts, p_i = c/n estimates
pass@1. With f_K(p) = 1 - (1-p)^K:

  R0    = base-reachable set: tasks whose *empirical* base pass@K (unbiased
          estimator over the n rollouts) lies strictly in (low, high)
  L     = sum over R0 of [f_K(p_base) - f_K(p_sft)]_+   (coverage destroyed)
  G     = sum over R0 of [f_K(p_sft) - f_K(p_base)]_+   (coverage created)
  rho_K = L / G   (inf if G == 0)

rho_K > 1 means standard SFT destroyed more base-reachable coverage than it
created -- the regime where TailSFT reliably preserved or improved coverage
in the paper's experiments.

Example:
  uv run python rl_data/scripts/sft/coverage_ratio_diagnostic.py \\
      --base-dir /weka/oe-adapt-default/pradeepd/tmax_solutions/tasks_rl_15k \\
      --sft-dir  /weka/oe-adapt-default/pradeepd/tmax_solutions/tasks_rl_15k \\
      --base-tag hosted_vllm_qwen35-9b-base \\
      --sft-tag  hosted_vllm_qwen35-9b-glm52-all-sft \\
      --ks 8 16 32
"""

import argparse
import json
from math import comb
from pathlib import Path


def f_k(p: float, k: int) -> float:
    """pass@k under independent sampling at per-sample success rate p."""
    return 1.0 - (1.0 - p) ** k


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from c successes in n samples (n >= k)."""
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load_counts(root: Path, tag: str) -> dict[str, tuple[int, int]]:
    """Map task name -> (num_success, num_runs) from matching summaries."""
    counts: dict[str, tuple[int, int]] = {}
    for task_dir in sorted(root.glob("task_*")):
        matches = sorted((task_dir / "solutions").glob(f"{tag}*_summary.json"))
        if not matches:
            continue
        if len(matches) > 1:
            raise SystemExit(
                f"{task_dir.name}: tag {tag!r} matches multiple summaries: "
                f"{[m.name for m in matches]}; use a more specific --*-tag"
            )
        summary = json.loads(matches[0].read_text())
        counts[task_dir.name] = (summary["num_success"], summary["num_runs"])
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base-dir", type=Path, required=True, help="corpus/summary-cache dir for the base model run")
    ap.add_argument("--sft-dir", type=Path, required=True, help="corpus/summary-cache dir for the SFT model run")
    ap.add_argument("--base-tag", required=True, help="summary filename prefix for the base model, e.g. hosted_vllm_qwen35-9b-base")
    ap.add_argument("--sft-tag", required=True, help="summary filename prefix for the SFT model")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32], help="pass@k scales to report (RL group size first)")
    ap.add_argument("--low", type=float, default=0.05, help="lower base-reachability bound on base pass@k")
    ap.add_argument("--high", type=float, default=0.95, help="upper base-reachability bound on base pass@k")
    ap.add_argument("--top", type=int, default=10, help="how many top L/G contributor tasks to print")
    args = ap.parse_args()

    base = load_counts(args.base_dir, args.base_tag)
    sft = load_counts(args.sft_dir, args.sft_tag)
    common = sorted(set(base) & set(sft))
    if not common:
        raise SystemExit("no tasks with summaries for both models")
    print(f"tasks: base={len(base)}  sft={len(sft)}  common={len(common)}")

    n_values = {base[t][1] for t in common} | {sft[t][1] for t in common}
    print(f"rollouts per task (n): {sorted(n_values)}")

    for k in args.ks:
        rows = []  # (task, p_base, p_sft, delta_fk)
        agg_base = agg_sft = 0.0
        for t in common:
            cb, nb = base[t]
            cs, ns = sft[t]
            if nb < k or ns < k:
                raise SystemExit(f"{t}: n < k ({nb},{ns} < {k}); rerun with more rollouts")
            agg_base += unbiased_pass_at_k(nb, cb, k)
            agg_sft += unbiased_pass_at_k(ns, cs, k)
            # base-reachable set membership: empirical base pass@k
            if not (args.low < unbiased_pass_at_k(nb, cb, k) < args.high):
                continue
            p_b, p_s = cb / nb, cs / ns
            rows.append((t, p_b, p_s, f_k(p_s, k) - f_k(p_b, k)))

        L = sum(-d for _, _, _, d in rows if d < 0)
        G = sum(d for _, _, _, d in rows if d > 0)
        rho = float("inf") if G == 0 else L / G
        print(f"\n=== k={k} ===")
        print(f"aggregate pass@{k}: base={agg_base / len(common):.3f}  sft={agg_sft / len(common):.3f}")
        print(f"base-reachable |R0|={len(rows)}/{len(common)}")
        print(f"L (coverage destroyed) = {L:.3f}")
        print(f"G (coverage created)   = {G:.3f}")
        print(f"rho_{k} = {rho:.3f}" + ("  -> TailSFT-favorable regime (rho > 1)" if rho > 1 else ""))

        rows.sort(key=lambda r: r[3])
        header = f"{'task':<40} {'p_base':>7} {'p_sft':>7} {'d_f' + str(k):>8}"
        losses = [r for r in rows if r[3] < 0][: args.top]
        gains = [r for r in reversed(rows) if r[3] > 0][: args.top]
        if losses:
            print(f"\ntop coverage losses (SFT un-taught these):\n{header}")
            for t, pb, ps, d in losses:
                print(f"{t:<40} {pb:>7.3f} {ps:>7.3f} {d:>8.3f}")
        if gains:
            print(f"\ntop coverage gains:\n{header}")
            for t, pb, ps, d in gains:
                print(f"{t:<40} {pb:>7.3f} {ps:>7.3f} {d:>8.3f}")


if __name__ == "__main__":
    main()
