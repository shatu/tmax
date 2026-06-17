# Agent Harnesses Compared

This document compares the agent harnesses available in `tmax` for driving a
model through terminal/coding tasks (used both for eval and for RL/SFT data
generation). It explains what each one is, how its agent loop works, and—most
usefully—how they differ from each other and from a textbook **ReAct** agent.

For *how to launch* these agents (CLI flags, providers, parsers, datasets) see
[`running_evals.md`](running_evals.md) §7. This doc is about *what they are and
why they differ*.

---

## TL;DR

Almost every agent here is a variant of the same idea: a **ReAct loop with one
`bash` tool**. The model thinks (THOUGHT), emits an action (a shell command),
sees the observation (truncated stdout/stderr), and repeats until it runs
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. What actually differs between them
is the *operational envelope* around that loop:

- **Where the loop runs** — host-side (Python driving `environment.exec()`) vs.
  inside the sandbox (an installed agent binary).
- **How actions are parsed** — native structured tool-calls vs. text/XML/JSON
  blocks vs. tmux keystrokes.
- **How shell state is kept** — stateless subshells, a save/restore wrapper, or
  a real persistent tmux session.
- **How context is managed** — let it grow until OOM, or proactively summarize.
- **How rich the tool surface is** — bash-only vs. bash+view/edit/submit.

| Agent | Kind | Runs | Tools | Action format | State | Context mgmt |
|---|---|---|---|---|---|---|
| **TassieAgent** | standalone `BaseAgent` | host-side | bash | structured tool-call | save/restore wrapper | none (grows → stop) |
| **TassumAgent** | TassieAgent subclass-style | host-side | bash | structured tool-call | save/restore wrapper | **proactive summarization** |
| **Vanillux2Agent** | standalone `BaseAgent` | host-side | bash | structured tool-call | save/restore wrapper | none |
| **VanilluxAgent** ⚠️ | SWE-agent wrapper | in sandbox | bash + view/edit/submit | SWE-agent internal | persistent (in-sandbox) | none (cache_control off) |
| **mini-swe-agent** | installed agent | in sandbox | bash | text ` ```bash ``` ` block | persistent shell | none |
| **terminus-2** | built-in `BaseAgent` | host-side | tmux keystrokes | JSON or XML plain | **persistent tmux** | proactive summarization |
| **swe-agent** | installed agent | in sandbox | bash + view/edit/submit | SWE-agent internal | persistent (in-sandbox) | history processors |

⚠️ `VanilluxAgent` is **broken against the pinned harbor 0.6.6** (imports
`ExecInput`, which doesn't exist in that build). Use `Vanillux2Agent` instead.

---

## The ReAct baseline (reference point)

A "simple ReAct-style agent" (Yao et al. 2022) is:

1. A system prompt that asks the model to alternate **Thought → Action →
   Observation**.
2. A **typed action vocabulary** (e.g. `search[query]`, `lookup[term]`,
   `finish[answer]`), parsed out of free-form text with a regex.
3. A loop that runs the parsed action, appends the observation, and repeats.
4. Termination via a dedicated `finish[]` action or a step cap.
5. Minimal-to-no error handling, output truncation, or context management.

Every agent below keeps the Thought→Action→Observation spine but changes one or
more of: the action *channel*, the action *vocabulary*, the *state* model, the
*context* model, and the *operational hardening*. The recurring theme in `tmax`
is the **mini-swe-agent thesis**: collapse the action vocabulary to a single
`bash` tool and put all the structure into the *prompt*, not into tool
machinery.

---

## The two architectural families

### A. Host-side litellm agents (`BaseAgent`)
`TassieAgent`, `TassumAgent`, `Vanillux2Agent`. The Python agent loop runs on
the **host**; only the bash commands cross into the sandbox via
`environment.exec()`. They call the model directly through litellm with
**structured (OpenAI-style) tool-calls**, so on Qwen3.5 they require
`--tool-call-parser qwen3_xml` (with the default `hermes`, tool-calls are
silently dropped and the agent loops on "Format error"). These three share an
almost identical implementation; the differences are small and listed below.

### B. In-sandbox installed agents
`mini-swe-agent`, `swe-agent`, and `VanilluxAgent` (a wrapper around
`swe-agent`). The agent binary is installed *inside* the task container and runs
there; harbor converts its native trajectory to ATIF for reporting. `tmax` does
not control their loop directly—it configures them via flags.

### C. terminus-2 (built-in, host-side, tmux)
Harbor's own built-in agent. Host-side like family A, but instead of a `bash`
tool it drives a **persistent tmux session** with raw keystrokes, and parses
the model's output as a plain **JSON or XML** document rather than a tool-call.

---

## Per-agent detail

### TassieAgent — the minimalist baseline
*Source:* `TassieAgent/agent.py` · *Default model:* `anthropic/claude-haiku-4-5`

A clean Harbor `BaseAgent` implementing a single-`bash`-tool ReAct loop:

- **Prompt:** one of two fixed system prompts (`SYSTEM_PROMPT_STATELESS` /
  `SYSTEM_PROMPT_PERSISTENT`) depending on `persistent_bash`; the user turn is
  the bare task. Both prompts mandate "a THOUGHT section … followed by exactly
  one bash command" and the `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` submit
  convention.
- **Action channel:** native structured tool-call—reads
  `response.tool_calls[0].function.arguments["command"]` (JSON). No text
  parsing.
- **State:** in persistent mode, every command is wrapped to restore `cwd`+`env`
  from `/tmp/.tassie/` before running and re-save after, so `cd`/`export`
  persist across turns without a long-lived shell process.
- **Observation:** each tool result truncated to 10 000 chars (head/tail with an
  "elided" marker).
- **Termination:** submit marker, `max_steps` (default 30), optional
  `cost_limit`, no-tool-call, or context-window overflow → stop.
- **Hardening:** exponential-backoff retry (5 attempts, 2→32 s); auth /
  not-found / context-window / unsupported-param / permission errors abort
  immediately.
- **Context management:** *none*. History grows until the context window is
  exceeded, then the loop stops and submits whatever exists.

Tassie is the repo default for most direct/Slurm eval scripts and reproduces the
legacy SFT data format byte-for-byte. Its known weakness: **no edit primitive**—
every code change is a full-file heredoc rewrite, which burns tokens and is
regression-prone on long tasks (this is the leading explanation for its ~8 pp
deficit vs. VanilluxAgent at a fixed budget).

### TassumAgent — Tassie + proactive context summarization
*Source:* `TassumAgent/agent.py`

Byte-for-byte the same loop, prompts, tools, parsing, truncation, budgets, and
retry logic as Tassie. The **one** difference is context management for
long-horizon tasks:

- Estimates token count (`chars/4`) each step; if free tokens
  (`max_input_tokens` − used, default `max_input_tokens=32768`) drop below
  `proactive_summarization_threshold` (default 8000), it summarizes.
- **4-step summarization cascade:** (1) *unwind*—drop the most-recent messages,
  always keeping system+task, until there's budget; (2) *standard*—two LLM calls
  (summarize → Q&A on open questions) compressed into a synthetic
  user+assistant pair, then re-append the kept tail; (3) *fallback*—a single
  shorter summary call if step 2 throws; (4) *ultimate fallback*—continue with
  just system+task+tail.
- Also triggers an **emergency** summarization if a `ContextWindowExceededError`
  is hit mid-run (Tassie just stops there).
- Gated by `enable_summarize` (default False per the eval docs—must be turned
  on).

Use Tassum when tasks are long enough to blow the context window; otherwise it's
strictly more LLM calls than Tassie for the same result.

### Vanillux2Agent — mini-swe-agent prompts, host-side
*Source:* `Vanillux2Agent/agent.py` · prompts vendored in
`rl_data/generator/vanillux_prompts.yaml`

The host-side litellm port of the `rl_data` vanillux solver. Functionally very
close to Tassie, but:

- **Prompts are the vendored mini-swe-agent v2.2 templates** (system + instance
  with the explicit 5-step "Recommended Workflow" and command examples), not
  Tassie's hand-written prompt.
- Higher budgets tuned to mini-swe-agent: `max_steps=64`, `max_tokens=16384`.
- Explicit **format-error recovery loop**: a malformed turn is fed back as an
  error message (up to `max_format_errors=64`) so the model self-corrects rather
  than wasting a step.
- Same save/restore persistent shell (`/tmp/.vanillux2/`), same 10 k head/tail
  truncation, same backoff retry.

This is the recommended agent on harbor 0.6.6 when you want the mini-swe-agent
recipe but a host-side loop you control. See the prior writeup in
[`running_evals.md`](running_evals.md) for the full Vanillux2 vs ReAct
breakdown.

### VanilluxAgent — SWE-agent wrapper (⚠️ currently broken)
*Source:* `VanilluxAgent/agent.py`

Not a loop of its own—a thin subclass of harbor's installed `SweAgent` that
patches the launch command:

- Raises budgets (`per_instance_cost_limit=10`, configurable
  `per_instance_call_limit` via `VANILLUX_CALL_LIMIT`).
- Disables the `cache_control` history processor (it crashes Gemini + litellm
  with "Missing corresponding tool call for tool response").
- Works around two harbor SWE-agent bugs (unset `CONDA_DEFAULT_ENV` under
  `set -u`; broken `$(pwd)` repo-path resolution) by pre-exporting and using a
  `preexisting` repo config with an `/app` symlink.

The actual loop, the **4-tool** surface (bash + view/edit/submit), prompts, and
parsing all come from upstream SWE-agent running *inside* the sandbox. It scores
higher than Tassie largely because of the `str_replace_editor` tool. **Caveat:**
it imports `ExecInput`, which is absent from the pinned harbor 0.6.6, so it fails
at import on every trial until the pin is bumped.

### mini-swe-agent — the upstream minimalist
*Source:* harbor built-in (`harbor/agents/installed/mini_swe_agent.py`); prompts
vendored into `rl_data/generator/vanillux_prompts.yaml`

The reference implementation of the "prompts > tools" philosophy: a **single
bash tool**, no editor/submit/review machinery, ~74 % on SWE-bench-verified
beating SWE-agent's own ~65 %. Installed inside the sandbox.

- **Action channel:** upstream parses a ` ```bash … ``` ` **text block** out of
  free-form output—so it is **independent of the tool-call parser** (any parser,
  even `hermes`, is fine). This is the key practical difference from the `tmax`
  litellm agents, which vendored these prompts but switched to native
  tool-calls.
- Persistent shell, head/tail truncation, format-error re-prompt, submit-marker
  termination—all the patterns the `tmax` agents copied.
- **Gotcha:** don't set `MSWEA_API_KEY`; mini-swe-agent forwards only that and
  skips `OPENAI_API_KEY`, breaking litellm credentials.

### terminus-2 — harbor's tmux agent
*Source:* harbor built-in (`harbor/agents/terminus_2/`)

The structurally most different agent here. Host-side, but instead of a `bash`
tool it operates a **persistent tmux session** and the model emits a plain
**JSON or XML** document (pluggable parser) with fields `analysis`, `plan`,
`commands`, `task_complete`. Each command is `keystrokes` + a `duration`,
letting it handle interactive/long-running TUI programs (send `C-c`, wait N
seconds, etc.)—something none of the bash-tool agents can do cleanly.

- **State:** a real long-lived tmux session—the strongest persistence model
  (genuine interactive shell, not a save/restore wrapper).
- **Context:** built-in proactive summarization (threshold default 8000), with
  optional linear-history segmentation.
- **Termination:** `task_complete: true` in the JSON, cost limit, or
  context-window overflow; `max_turns` defaults effectively unlimited.
- Supports extended thinking (`max_thinking_tokens`) and full ATIF trajectories.

Used by the RL-data generation scripts (`run_rldata_claude.sh`) to produce
training traces; those traces assume a persistent shell, which is why the
inference harnesses also use persistent state.

### swe-agent — upstream SWE-agent
The full upstream agent (Yang et al. 2024) installed in the sandbox: bash +
view/edit/submit tools, its own history processors, internal string parsing.
`VanilluxAgent` is just this with patched limits. Uses `hosted_vllm/` +
`hermes`.

---

## Dimension-by-dimension comparison

### 1. Action channel (how the model's action is parsed)
- **Native structured tool-call** (Tassie, Tassum, Vanillux2): robust, but ties
  you to a model+parser combo. On Qwen3.5 you *must* use `qwen3_xml`.
- **Text bash block** (mini-swe-agent): parser-independent, simplest to serve.
- **JSON/XML plain document** (terminus-2): richest action schema (keystrokes +
  duration + plan + completion flag), supports interactive programs.
- **SWE-agent internal** (swe-agent, VanilluxAgent): multi-tool, opaque to
  `tmax`.
- **vs ReAct:** ReAct parses a typed `Action: verb[arg]` line with a regex. All
  of these move the action onto a more structured or more general channel.

### 2. Tool surface
- **bash-only:** Tassie, Tassum, Vanillux2, mini-swe-agent, terminus-2 (as
  keystrokes). Editing is done via `sed`/heredoc.
- **bash + view/edit/submit:** swe-agent, VanilluxAgent. The dedicated editor is
  the main measured quality lift on long tasks.
- **vs ReAct:** ReAct typically has 3–10 typed tools; the `tmax` trend is the
  opposite—*fewer* tools, richer prompts.

### 3. Shell-state model
- **Stateless subshell:** optional in Tassie/Tassum (each command fresh; must
  `cd /path && …`).
- **Save/restore wrapper:** Tassie/Tassum/Vanillux2 persistent mode—snapshot
  `pwd`+`export -p` to `/tmp/.<agent>/` around each command. State persists
  without a live shell process.
- **Genuine persistent session:** mini-swe-agent (login shell), swe-agent
  (in-sandbox), terminus-2 (tmux). terminus-2's tmux is the only one that
  supports true interactivity.
- **vs ReAct:** ReAct actions are usually stateless function calls; persistence
  here is a deliberate addition for coding tasks.

### 4. Context management
- **None (grow until OOM):** Tassie, Vanillux2, mini-swe-agent, swe-agent
  (modulo its history processors).
- **Proactive summarization:** Tassum (4-step cascade), terminus-2 (built-in).
- **vs ReAct:** vanilla ReAct has no context management; this is the single
  biggest "production" addition for long-horizon tasks.

### 5. Observation truncation
All `tmax` agents truncate per-observation to ~10 k chars (head 5 k + tail 5 k +
hint). ReAct usually hard-cuts or drops. Keeping both head and tail preserves
e.g. the error line at the end of a stack trace.

### 6. Termination
- **Shell submit sentinel** (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`):
  Tassie, Tassum, Vanillux2, mini-swe-agent. "Done" travels through the same
  bash channel as everything else—no dedicated finish action.
- **Completion flag:** terminus-2 (`task_complete: true`), swe-agent (submit
  tool).
- Plus universal caps: step/turn limit, cost limit, context overflow.
- **vs ReAct:** ReAct uses a dedicated `finish[]` action.

### 7. Budgets (defaults; launch scripts override)
- Tassie/Tassum: `max_steps=30` step limit. Vanillux2: `max_steps=64`.
- VanilluxAgent: per-instance *call* limit (`VANILLUX_CALL_LIMIT`).
- terminus-2: effectively unlimited turns; bounded by cost.
- **Fairness note:** call-limit vs step-limit agents are *not* directly
  comparable; keep the harness fixed when comparing models.

### 8. Error/retry hardening
Family-A agents share exponential backoff (5 attempts, 2→32 s) with an
abort-immediately set (auth/not-found/context-window/unsupported/permission),
plus format-error recovery (Vanillux2 explicitly; mini-swe-agent re-prompts).
ReAct typically has minimal or no retry logic.

---

## Practical compatibility (harbor 0.6.6)

| Agent | Works on 0.6.6? | `--model-provider` | `--tool-call-parser` (Qwen3.5) |
|---|---|---|---|
| TassieAgent | yes | `openai` | **`qwen3_xml`** |
| TassumAgent | yes | `openai` | **`qwen3_xml`** |
| Vanillux2Agent | yes | `openai` | **`qwen3_xml`** |
| mini-swe-agent | yes | `openai` | any (parser-independent) |
| terminus-2 | yes | `openai` | `qwen3_xml` (safe) |
| swe-agent | yes | `hosted_vllm` | `hermes` |
| VanilluxAgent | **NO** (import error) | `hosted_vllm` | `hermes` |

Key traps (from [`running_evals.md`](running_evals.md) §7):
- Structured-tool agents on Qwen3.5 **must** use `qwen3_xml`; `hermes` silently
  drops their tool-calls → 0 useful steps. `qwen_xml` is not a valid name.
- Don't set `MSWEA_API_KEY` for built-in agents (breaks litellm credentials).
- `VanilluxAgent` is the stock `launch_eval.sh` default but fails to import on
  the pinned harbor—use `Vanillux2Agent` or `mini-swe-agent` until the pin is
  bumped.

---

## Which to use

- **Default eval / fast & simple:** `Vanillux2Agent` (works on 0.6.6,
  mini-swe-agent prompts, host-side loop you control) or `TassieAgent`.
- **Long-horizon tasks that blow the context window:** `TassumAgent` or
  `terminus-2`.
- **Interactive / TUI / long-running-program tasks:** `terminus-2` (tmux).
- **Best raw quality on edit-heavy SWE tasks:** `swe-agent` / `VanilluxAgent`
  (once the harbor pin is fixed)—the dedicated editor matters.
- **Parser-agnostic serving:** `mini-swe-agent` (text bash blocks, any parser).

---

*Sources: `TassieAgent/agent.py`, `TassumAgent/agent.py`,
`Vanillux2Agent/agent.py`, `VanilluxAgent/agent.py`,
`rl_data/generator/vanillux_solver.py`, `rl_data/generator/vanillux_prompts.yaml`,
harbor `agents/terminus_2/` and `agents/installed/mini_swe_agent.py`,
`docs/running_evals.md` §7, and the run-terminal-eval skill compat matrix.*
