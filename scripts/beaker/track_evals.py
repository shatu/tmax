#!/usr/bin/env python3
"""
track_evals.py -- idempotent tracker for terminal-bench (harbor) eval experiments.

Run with the open-instruct uv env (it has wandb), e.g.:
  cd /weka/nora-default/shashankg/code/open-instruct && \
    uv run python /weka/nora-default/shashankg/code/tmax/scripts/beaker/track_evals.py <subcmd> ...

Rows are keyed by `beaker_url` and UPSERTED into the CSV (default: this file's dir /
dppo9b_4n64k_tb21_evals.csv), so re-running flips `running`->`done` and never duplicates.

Subcommands:
  add       --exp ID [ID ...]           extract+upsert eval rows for the given experiment(s)
            [--model-name NAME] [--step N]
            [--require-tb21] [--require-k 5] [--require-maxlen 65536]
  discover  --workspace ai2/oe-agents --filter <substr> [--author shashankg]
            list matching eval experiments (does NOT modify csv); pipe ids into `add`
  refresh                                re-extract every row currently marked status=running
  train     --wandb ID [ID ...]         fill train_* / wandb_url / train_beaker_url for the
                                         native-checkpoint rows (matched by `step`), pulling
                                         per-step metrics across the resume-chain runs (earliest wins)

Config verification (`--require-*`) is recorded in the row regardless; the flags only gate
whether a mismatching experiment is skipped. See the `track-terminal-evals` skill for conventions.
"""
import argparse, csv, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "dppo9b_4n64k_tb21_evals.csv")
WANDB_PROJECT = "ai2-llm/oe-general-agents"

COLS = ["model_name","step","pass@1","pass@5","mean_reward","mean_reward_std","errors",
        "error_rate","agent_timeout_err","infra_err","pass1_infra_adj","n_trials","dataset",
        "k","max_model_len","workspace","beaker_url","status","train_grp_perf","train_grp_perf_w5",
        "train_scores","train_scores_w5","train_kl2","train_seq_len","wandb_url","train_beaker_url"]

# exception types that are NOT the model's reasoning failure -> "infra" bucket.
AGENT_ERR = {"AgentTimeoutError"}   # everything else (RuntimeError=vLLM/conn, Verifier/Reward/BadRequest...) = infra


def sh(cmd, timeout=200):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout


def exp_json(exp):
    return json.loads(sh(f"beaker experiment get {exp} --format json"))[0]


def spec_env(exp):
    d = json.loads(sh(f"beaker experiment spec {exp} --format json"))
    t = (d[0] if isinstance(d, list) else d)["tasks"][0]
    return {e["name"]: e.get("value") for e in t.get("envVars", [])}


def parse_scores(log):
    p1 = re.search(r"pass@1:\s+([0-9.]+)", log)
    p5 = re.search(r"pass@5:\s+([0-9.]+)", log)
    mr = re.search(r"Mean reward:\s+([0-9.]+) \+/- ([0-9.]+)", log)
    exc = dict(re.findall(r"[│]\s*([A-Za-z]+Error)\s*[│]\s*([0-9]+)\s*[│]", log))
    if not p1:
        return None
    return {
        "pass@1": p1.group(1),
        "pass@5": p5.group(1) if p5 else "",
        "mean_reward": mr.group(1) if mr else "",
        "mean_reward_std": mr.group(2) if mr else "",
        "exc": {k: int(v) for k, v in exc.items()},
    }


def extract_eval(exp, model_name=None, step=None):
    ej = exp_json(exp)
    ev = spec_env(exp)
    author = (ej.get("author") or {}).get("name", "")
    ws = ej.get("workspaceRef", {}).get("fullName", "")
    dpath = (ev.get("DATASET_PATH") or "")
    dataset = os.path.basename(dpath) if dpath else (ev.get("DATASET") or "")
    served = ev.get("SERVED_MODEL_NAME") or ""
    # find the job whose log carries the score summary
    sc = None
    for j in ej["jobs"]:
        log = sh(f"beaker job logs {j['id']} 2>&1 | tail -400")
        sc = parse_scores(log)
        if sc:
            break
    status = "done" if sc else ("running" if any(
        jj["status"].get("started") and not jj["status"].get("finalized") for jj in ej["jobs"]) else "no-score")
    n_trials = ""
    m = re.search(r"([0-9]+) trials", sh(f"beaker job logs {ej['jobs'][0]['id']} 2>&1 | tail -400")) if not sc else None
    row = {c: "" for c in COLS}
    row.update({
        "model_name": model_name or (ev.get("MODEL_PATH") or served or ""),
        "step": str(step) if step is not None else "",
        "dataset": dataset, "k": str(ev.get("N_ATTEMPTS") or ""),
        "max_model_len": str(ev.get("MAX_MODEL_LEN") or ""),
        "workspace": ws, "beaker_url": f"https://beaker.org/ex/{exp}", "status": status,
    })
    if sc:
        agent = sum(v for k, v in sc["exc"].items() if k in AGENT_ERR)
        infra = sum(v for k, v in sc["exc"].items() if k not in AGENT_ERR)
        tot = agent + infra
        nt = 445  # 89 tasks x k=5 (TB2.1); adjust if k differs
        try:
            nt = 89 * int(ev.get("N_ATTEMPTS") or 5)
        except Exception:
            pass
        p1 = float(sc["pass@1"])
        row.update({
            "pass@1": sc["pass@1"], "pass@5": sc["pass@5"], "mean_reward": sc["mean_reward"],
            "mean_reward_std": sc["mean_reward_std"], "errors": str(tot) if sc["exc"] else "",
            "n_trials": str(nt),
        })
        if sc["exc"]:
            row["error_rate"] = f"{tot/nt:.3f}"
            row["agent_timeout_err"] = str(agent)
            row["infra_err"] = str(infra)
            row["pass1_infra_adj"] = f"{(p1*nt)/(nt-infra):.4f}" if nt - infra > 0 else ""
    row["_author"] = author
    row["_dpath"] = dpath
    return row


def load_csv(path):
    if not os.path.exists(path):
        return [dict.fromkeys(COLS, "")][:0]
    return list(csv.DictReader(open(path)))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})


def upsert(rows, newrow):
    key = newrow["beaker_url"]
    for r in rows:
        if r.get("beaker_url") == key:
            for c in COLS:
                if newrow.get(c):           # only overwrite with non-empty (preserve train_* etc.)
                    r[c] = newrow[c]
            return rows, "updated"
    rows.append({c: newrow.get(c, "") for c in COLS})
    return rows, "added"


def cmd_add(a):
    rows = load_csv(a.csv)
    for exp in a.exp:
        r = extract_eval(exp, a.model_name, a.step)
        if a.require_tb21 and "terminal-bench-2-1" not in r["_dpath"]:
            print(f"SKIP {exp}: dataset {r['dataset']} != terminal-bench-2-1"); continue
        if a.require_k and str(a.require_k) != r["k"]:
            print(f"SKIP {exp}: k={r['k']} != {a.require_k}"); continue
        if a.require_maxlen and str(a.require_maxlen) != r["max_model_len"]:
            print(f"SKIP {exp}: max_model_len={r['max_model_len']} != {a.require_maxlen}"); continue
        rows, act = upsert(rows, r)
        print(f"{act.upper():7} {r['model_name']}{'/'+r['step'] if r['step'] else ''}  "
              f"p@1={r['pass@1']} p@5={r['pass@5']} infra={r['infra_err']} status={r['status']}  {exp}")
    write_csv(a.csv, rows)


def cmd_refresh(a):
    rows = load_csv(a.csv)
    for r in rows:
        if r.get("status") == "running" and r.get("beaker_url", "").startswith("http"):
            exp = r["beaker_url"].rsplit("/", 1)[-1]
            nr = extract_eval(exp, r.get("model_name") or None, r.get("step") or None)
            for c in COLS:
                if nr.get(c):
                    r[c] = nr[c]
            print(f"REFRESH {r['model_name']}/{r.get('step')}: status={r['status']} p@1={r['pass@1']}")
    write_csv(a.csv, rows)


def cmd_discover(a):
    d = json.loads(sh(f"beaker workspace experiments {a.workspace} --format json"))
    for e in d:
        au = (e.get("author") or {}).get("name", "")
        nm = e.get("name") or ""
        desc = e.get("description") or ""
        if a.author and au != a.author:
            continue
        if not (nm.startswith("eval-") or "Harbor eval" in desc):
            continue
        if a.filter and a.filter not in nm and a.filter not in desc:
            continue
        print(f"{e['id']}  {nm[:72]}")


def cmd_train(a):
    import wandb
    api = wandb.Api()
    KEYS = ["training_step", "val/avg_group_performance_pre_filter", "scores",
            "objective/kl2_avg", "val/sequence_lengths"]
    grp, sco, kl, sl, wb_for_step = {}, {}, {}, {}, {}
    wb2beaker = {}
    for rid in a.wandb:
        r = api.run(f"{WANDB_PROJECT}/{rid}")
        # find the training beaker experiment (its desc holds this wandb id)
        for row in load_csv(a.csv):
            pass
        for h in r.scan_history(keys=KEYS, page_size=5000):
            ts = h.get("training_step")
            if ts is None:
                continue
            ts = int(ts)
            if ts in wb_for_step:
                continue  # earliest run in --wandb order wins
            wb_for_step[ts] = rid
            for m, k in [(grp, "val/avg_group_performance_pre_filter"), (sco, "scores"),
                         (kl, "objective/kl2_avg"), (sl, "val/sequence_lengths")]:
                v = h.get(k)
                if isinstance(v, (int, float)):
                    m[ts] = v
    # resolve wandb id -> training beaker exp by scanning both workspaces
    for ws in ["ai2/oe-agents", "ai2/general-tool-use"]:
        for e in json.loads(sh(f"beaker workspace experiments {ws} --format json")):
            desc = e.get("description") or ""
            for rid in a.wandb:
                if rid in desc and rid not in wb2beaker:
                    wb2beaker[rid] = e["id"]

    def at(m, step):
        c = [s for s in m if abs(s - step) <= 3]
        return m[min(c, key=lambda x: abs(x - step))] if c else None

    def w5(m, step):
        v = [m[s] for s in m if abs(s - step) <= 5]
        return sum(v) / len(v) if v else None

    def f(v, nd=4):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else ""

    rows = load_csv(a.csv)
    for r in rows:
        st = r.get("step", "")
        if not st.isdigit():
            continue
        step = int(st)
        if not any(abs(s - step) <= 3 for s in grp):
            continue
        rid = wb_for_step.get(min((s for s in wb_for_step if abs(s - step) <= 3),
                                  key=lambda x: abs(x - step)), "")
        r["train_grp_perf"] = f(at(grp, step))
        r["train_grp_perf_w5"] = f(w5(grp, step))
        r["train_scores"] = f(at(sco, step))
        r["train_scores_w5"] = f(w5(sco, step))
        r["train_kl2"] = f(at(kl, step))
        r["train_seq_len"] = f(at(sl, step), 0)
        r["wandb_url"] = f"{'https://wandb.ai/'}{WANDB_PROJECT}/{rid}" if rid else r.get("wandb_url", "")
        if rid in wb2beaker:
            r["train_beaker_url"] = f"https://beaker.org/ex/{wb2beaker[rid]}"
        print(f"TRAIN {r['model_name']}/{step}: grp_w5={r['train_grp_perf_w5']} "
              f"kl2={r['train_kl2']} seq={r['train_seq_len']} run={rid}")
    write_csv(a.csv, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("--exp", nargs="+", required=True)
    a.add_argument("--model-name"); a.add_argument("--step")
    a.add_argument("--require-tb21", action="store_true")
    a.add_argument("--require-k", type=int); a.add_argument("--require-maxlen", type=int)
    d = sub.add_parser("discover"); d.add_argument("--workspace", required=True)
    d.add_argument("--filter", default=""); d.add_argument("--author", default="shashankg")
    sub.add_parser("refresh")
    t = sub.add_parser("train"); t.add_argument("--wandb", nargs="+", required=True)
    args = ap.parse_args()
    {"add": cmd_add, "discover": cmd_discover, "refresh": cmd_refresh, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
