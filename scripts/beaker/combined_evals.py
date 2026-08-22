#!/usr/bin/env python3
"""Directly build & maintain the ONE combined terminal-bench eval sheet.
Each row carries its tb21_beaker_url + tblite_beaker_url; this tool re-extracts scores
straight from those experiments' logs (no intermediate per-benchmark sheets).

  refresh [--force]                 fill scores for rows whose block has a beaker_url but no pass@1
                                    (or all, with --force). Idempotent; run after evals finish.
  add --model NAME [--step N] --max-len M --workspace WS
      [--tb21 EXP] [--tblite EXP]
      [--grp-w5 X --kl2 X --seq-len X --wandb URL --train-exp EXP]
                                    append/replace a row, then refresh it.

Run with the open-instruct uv env. TB2.1 = 89 tasks, TBlite (openthoughts-tblite) = 100 tasks;
n_trials is read from the log ("X/Y trials") when present. Agent-timeout = model; everything
else (RuntimeError/Verifier/etc.) = infra; pass1_adj excludes infra-failed trials.
"""
import argparse, csv, json, re, subprocess, sys, os

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terminalbench_combined_evals.csv")
COLS = ["model_name","step","max_model_len",
        "tb21_pass@1","tb21_pass@5","tb21_pass1_adj","tb21_err_rate","tb21_beaker_url",
        "tblite_pass@1","tblite_pass@5","tblite_pass1_adj","tblite_err_rate","tblite_beaker_url",
        "train_grp_perf_w5","train_kl2","train_seq_len","wandb_url","train_beaker_url","workspace"]
DEFAULT_TASKS = {"tb21": 89, "tblite": 100}

ORDER = [("hamishivi/Qwen3-8B",""),("Qwen/Qwen3.5-2B",""),("Qwen/Qwen3.5-4B",""),("Qwen/Qwen3.5-9B",""),
         ("allenai/tmax-sft-8b",""),("sft-tmax-qwen3-8b-1ep",""),("sft-tmax-qwen3-8b-2ep",""),
         ("sft-tmax-qwen35-9b-1ep",""),("sft-tmax-qwen35-9b-2ep",""),
         ("sft-tmax-qwen35-9b-big-1ep",""),("sft-tmax-qwen35-9b-big-2ep",""),
         ("allenai/tmax-2b",""),("allenai/tmax-4b",""),("allenai/tmax-9b","")] + \
        [("swerl-qwen35-9b-dppo-4n64k",str(s)) for s in range(20,1001,20)]
OIDX = {k:i for i,k in enumerate(ORDER)}

def sh(c, t=200): return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout

def parse(exp):
    """Return (pass@1, pass@5, agent_err, infra_err, n_trials) or None if no score yet."""
    try: js = json.loads(sh(f"beaker experiment get {exp} --format json"))[0]["jobs"]
    except Exception: return None
    for j in js:
        log = sh(f"beaker job logs {j['id']} 2>&1 | tail -400")
        m1 = re.search(r"pass@1:\s+([0-9.]+)", log)
        if not m1: continue
        p5 = dict((int(n), v) for n, v in re.findall(r"pass@(\d+):\s+([0-9.]+)", log)).get(5, "")
        exc = {a: int(b) for a, b in re.findall(r"[│]\s*([A-Za-z]+Error)\s*[│]\s*([0-9]+)\s*[│]", log)}
        nt = re.search(r"([0-9]+)/([0-9]+) trials", log)
        prog = re.findall(r"errors=([0-9]+)", log)  # progress-line total (fallback when no exception table)
        if exc:
            agent = exc.get("AgentTimeoutError", 0); infra = sum(v for k, v in exc.items() if k != "AgentTimeoutError")
            return m1.group(1), p5, agent, infra, (int(nt.group(2)) if nt else None), True
        # no exception table: use progress-line total; can't split agent/infra
        tot = int(prog[-1]) if prog else 0
        return m1.group(1), p5, None, tot, (int(nt.group(2)) if nt else None), False
    return None

def fill(row, block, force):
    url = row.get(f"{block}_beaker_url", "")
    if not url.startswith("http"): return
    if row.get(f"{block}_pass@1") and not force: return
    r = parse(url.rsplit("/", 1)[-1])
    if not r: print(f"  {row['model_name']}/{row['step']} {block}: no score yet"); return
    p1, p5, agent, infra, nt, has_table = r
    nt = nt or DEFAULT_TASKS[block] * 5
    tot = (agent or 0) + infra
    row[f"{block}_pass@1"] = p1
    row[f"{block}_pass@5"] = p5
    row[f"{block}_err_rate"] = f"{tot/nt:.3f}"
    # infra-adjusted pass@1 only meaningful when we can split off infra errors (exception table present)
    row[f"{block}_pass1_adj"] = (f"{(float(p1)*nt)/(nt-infra):.4f}" if (has_table and nt-infra > 0) else "")
    print(f"  {row['model_name']}/{row['step']} {block}: p@1={p1} p@5={p5 or '-'} err={tot}/{nt}{'' if has_table else ' (no table)'}")

def load(): return list(csv.DictReader(open(CSV))) if os.path.exists(CSV) else []
def save(rows):
    rows.sort(key=lambda r: OIDX.get((r["model_name"], r["step"]), 10**6))
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow({c: r.get(c, "") for c in COLS})

def cmd_refresh(a):
    rows = load()
    for r in rows:
        fill(r, "tb21", a.force); fill(r, "tblite", a.force)
    save(rows); print(f"refreshed {len(rows)} rows -> {CSV}")

def cmd_add(a):
    rows = load()
    key = (a.model, a.step or "")
    row = next((r for r in rows if (r["model_name"], r["step"]) == key), None)
    if row is None:
        row = {c: "" for c in COLS}; rows.append(row)
    row.update({"model_name": a.model, "step": a.step or "", "max_model_len": a.max_len or row.get("max_model_len",""),
                "workspace": a.workspace or row.get("workspace","")})
    for k, v in [("tb21_beaker_url", a.tb21 and f"https://beaker.org/ex/{a.tb21}"),
                 ("tblite_beaker_url", a.tblite and f"https://beaker.org/ex/{a.tblite}"),
                 ("train_grp_perf_w5", a.grp_w5), ("train_kl2", a.kl2), ("train_seq_len", a.seq_len),
                 ("wandb_url", a.wandb), ("train_beaker_url", a.train_exp and f"https://beaker.org/ex/{a.train_exp}")]:
        if v: row[k] = v
    fill(row, "tb21", True); fill(row, "tblite", True)
    save(rows); print(f"added/updated {a.model}/{a.step or ''}")

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("refresh"); rp.add_argument("--force", action="store_true")
    ad = sub.add_parser("add")
    ad.add_argument("--model", required=True); ad.add_argument("--step", default="")
    ad.add_argument("--max-len"); ad.add_argument("--workspace")
    ad.add_argument("--tb21"); ad.add_argument("--tblite")
    ad.add_argument("--grp-w5"); ad.add_argument("--kl2"); ad.add_argument("--seq-len")
    ad.add_argument("--wandb"); ad.add_argument("--train-exp")
    a = ap.parse_args()
    {"refresh": cmd_refresh, "add": cmd_add}[a.cmd](a)

main()
