# Sentinel: an autonomous goal-driven agent

Sentinel takes a goal in plain language and works toward it on its own. It looks at its environment, makes a plan, uses tools to act, checks the results, recovers from errors and keeps going until the goal is met. Before the run can end, an independent reviewer has to accept the evidence. After the run, Sentinel writes the lessons it learned to long-term memory for next time.

Every decision comes from an LLM. It runs **for free** on Google Gemini's free tier, Groq's free tier, or a local Ollama model, and also on Claude through the Anthropic API. The code supplies the loop, the tools, the memory and the safety limits. There are no scripted answers and no hard-coded task logic.

<sub>Illustrative, abridged console output:</sub>

```
$ sentinel run "Make all tests in examples/buggy_inventory pass without editing the tests"

◉ perceive  environment snapshot taken
✻ reason    The project has a tests/ folder; run the suite first to see what fails...
☰ plan      1. [>] run the test suite   2. [ ] read inventory.py   3. [ ] fix each bug   4. [ ] re-run
▶ act       run_command {"command": "python -m pytest examples/buggy_inventory -q"}
  ◂ [stdout] 6 failed, 1 passed ...
▶ act       read_file {"path": "examples/buggy_inventory/inventory.py"}
▶ act       edit_file {...}
◉ observe   modified: examples/buggy_inventory/inventory.py
...
✔ verify    reviewer accepted: test output shows 7 passed
✎ reflect   lessons saved to memory: • Tests here run with python -m pytest <dir> -q ...
```

---

## Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Demo goals](#demo-goals)
4. [How it works](#how-it-works)
5. [Safety and guardrails](#safety-and-guardrails)
6. [Project layout](#project-layout)
7. [Testing](#testing)
8. [Design decisions and limits](#design-decisions-and-limits)

---

## Installation

**Requirements:** Python 3.10 or newer, and an API key from one LLM provider:

| Provider | Cost | Get a key | Set |
|---|---|---|---|
| **Google Gemini** (recommended free option) | free tier, no credit card | https://aistudio.google.com/apikey | `GEMINI_API_KEY` |
| **Groq** | free tier, no credit card (tight per-minute token limits) | https://console.groq.com/keys | `GROQ_API_KEY` |
| **Ollama** (local) | free, runs on your machine; needs a strong machine and a tool-calling model | https://ollama.com | none: `--provider ollama --model <name>` |
| **Anthropic Claude** | paid, best results | https://console.anthropic.com | `ANTHROPIC_API_KEY` |

Sentinel picks the provider from whichever key is set, or you can choose with `--provider`. On free providers it picks a suitable model automatically from the provider's live model list. Free tiers have per-minute rate limits; Sentinel waits and retries when it hits one, so a run may pause briefly.

```bash
# 1. Clone
git clone <this-repo-url> sentinel-agent
cd sentinel-agent

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 3. Install (adds the `sentinel` command)
pip install -e ".[dev]"
#   or: pip install -r requirements.txt   (then use `python -m sentinel` instead of `sentinel`)

# 4. Provide your API key (example: free Gemini key), either in a .env file in the directory you run from...
cp .env.example .env                         # then edit .env and set GEMINI_API_KEY=...
# ...or as an environment variable:
export GEMINI_API_KEY=...                    # macOS/Linux
setx GEMINI_API_KEY ...                      # Windows (open a new terminal afterwards)

# 5. Check that everything works (no API key needed for the tests)
python -m pytest
```

## Usage

```bash
sentinel run "<goal>" [options]      # pursue one goal until it is done
sentinel chat [options]              # enter goals one after another; memory carries over
sentinel memory list|search|clear    # look at what the agent remembers
```

| Option | Default | Meaning |
|---|---|---|
| `-w, --workspace DIR` | current dir | Directory the agent may read and change. Every file operation and command is confined to it. |
| `--approval auto\|risky\|all` | `risky` | `auto`: fully autonomous. `risky`: asks you before shell or code execution. `all`: asks before every side effect. |
| `--max-steps N` | 40 | Hard cap on reasoning/acting steps. |
| `--provider` | `auto` | `gemini`, `groq`, `ollama`, `anthropic`, or `openai-compatible` (any OpenAI-style endpoint, with `--base-url`). `auto` picks from the key you have set. |
| `--model` | auto | Model ID. Defaults to `claude-opus-5-5` on Anthropic; on free providers it is picked from the live model list. |
| `--budget USD` | 3.0 | Anthropic only: hard cap on API spend per goal (`0` = no cap). Free providers cost nothing. |
| `--effort` | `high` | Anthropic only: reasoning depth (`low`, `medium`, `high`, `xhigh`, `max`). |
| `--no-web` | off | Anthropic only: turn off the server-side web search tool. Every provider can still read pages with `fetch_url`. |
| `--no-fallback` | off | Anthropic only: turn off the server-side refusal fallback (see below). |
| `-q, --quiet` | off | Hide reasoning summaries and tool output. |

`python -m sentinel ...` works the same as `sentinel ...`.

Exit codes: `0` success, `1` the goal was not achieved (failed, partial or stopped), `2` a configuration error such as a missing API key.

## Demo goals

The `examples/` folder contains two small environments. Run these from the repository root:

```bash
# 1. Debugging: six failing tests from seeded bugs, found and fixed by running the tests and reading the code
sentinel run "Make all tests in examples/buggy_inventory pass. Fix the code, not the tests." --approval auto

# 2. Data analysis: explores an unfamiliar CSV, computes results, writes a report and checks it
sentinel run "Analyse examples/sales/sales.csv: revenue by product and by month, the fastest growing and fastest declining product, and write the findings with a small table to reports/sales_summary.md" --approval auto

# 3. Research: web search plus fetching pages, with sources cited (web search needs the Anthropic provider;
#    on free providers the agent reads pages with fetch_url instead)
sentinel run "Find the current stable Python release and its three most notable new features; write notes/python_release.md with links to the sources"

# 4. Memory across runs: after run 1, this run starts with the lessons it saved
sentinel run "Add a test for Inventory.total_value with an empty inventory and make sure the suite still passes" --approval auto
sentinel memory list
```

To reset the demo project after a run: `git checkout examples/`.

## How it works

```
             ┌──────────────────────────────────────────────────────────┐
  goal ───▶  │ PERCEIVE  workspace snapshot · platform · git state      │
             │           + relevant memories from earlier runs          │
             └───────────────┬──────────────────────────────────────────┘
                             ▼
             ┌──────────────────────────────────────────────────────────┐
         ┌─▶ │ REASON    Claude reads the history, keeps a plan         │
         │   │           (update_plan) and chooses the next tool calls  │
         │   └───────────────┬──────────────────────────────────────────┘
         │                   ▼
         │   ┌──────────────────────────────────────────────────────────┐
         │   │ ACT       validate input → guardrails → human approval   │
         │   │           (if the policy asks) → run the tool            │
         │   └───────────────┬──────────────────────────────────────────┘
         │                   ▼
         │   ┌──────────────────────────────────────────────────────────┐
         └── │ OBSERVE   tool results + detected file changes;          │
             │           3 failed steps in a row → "step back, re-plan" │
             └───────────────┬──────────────────────────────────────────┘
                             ▼ agent calls finish(summary, evidence)
             ┌──────────────────────────────────────────────────────────┐
             │ VERIFY    an independent reviewer checks the evidence    │
             │           against the goal and the action log;           │
             │           rejected → feedback goes back to REASON        │
             └───────────────┬──────────────────────────────────────────┘
                             ▼ accepted
             ┌──────────────────────────────────────────────────────────┐
             │ REFLECT   lessons + episode → long-term memory           │
             └──────────────────────────────────────────────────────────┘
```

### Perception (`sentinel/perception.py`)
- **Initial snapshot:** OS and shell, Python version, time, workspace tree and `git status`, plus the long-term memories most relevant to the goal.
- **Change detection after every step:** a watcher compares file mtimes and sizes and tells the agent what was added, modified or deleted. The agent sees side effects it didn't directly cause, such as a script that wrote an output file.
- **Active perception:** the agent observes more on demand with `list_files`, `read_file`, `search_files`, `fetch_url` and web search.

### Reasoning and planning (`sentinel/agent.py`, `sentinel/prompts.py`)
- Claude runs with **adaptive extended thinking**. A summary of its reasoning is shown live as `✻ reason`.
- The agent keeps an explicit **plan** with the `update_plan` tool: steps with `pending / in_progress / done / blocked / skipped` statuses, revised as it learns.
- The system prompt sets out working principles, not task rules: ground decisions in observations, check effects, diagnose failures instead of retrying blindly, and ask a human only when truly blocked.

### Action (`sentinel/tools/`)
| Tool | Risk | Purpose |
|---|---|---|
| `list_files`, `read_file`, `search_files` | read | Observe the workspace |
| `write_file`, `edit_file` | write | Change files (exact-match edits) |
| `run_command`, `run_python` | exec | Tests, builds, CLIs, data analysis |
| `fetch_url` | network | Read web pages or call public JSON APIs |
| `web_search` | network | Anthropic's server-side web search |
| `update_plan`, `remember`, `recall`, `ask_human`, `finish` | control | Agent self-management |

The model may request several tools in one turn. Each call is validated against its schema before it runs, and errors go back to the model as `is_error` results so it can recover.

### Memory (`sentinel/memory.py`)
- **Short-term:** the full conversation (thinking, tool calls and results) plus the current plan. The conversation prefix is prompt-cached, so long runs stay cheap.
- **Long-term:** a JSONL store in `<workspace>/.sentinel/memory.jsonl` with three kinds of entry:
  - `fact`: saved by the agent with `remember`, e.g. "tests run with `python -m pytest -q`"
  - `lesson`: extracted automatically after each run by a reflection step
  - `episode`: a one-line record of each goal and its outcome
- Retrieval is TF-IDF-style keyword overlap with a recency tie-break. It needs no embedding service or vector database.

### Self-reflection and recovery
- **Error recovery:** failed tools come back as errors with a message. After three fully failed steps in a row, the agent is told to step back and re-plan.
- **Verification before stopping:** `finish` doesn't end the run by itself. A separate LLM call reviews the claim against the real action log and returns a structured verdict (`accept`, `goal_met`, `gaps`, `feedback`). If the evidence doesn't hold up, the agent gets the gaps and keeps working. After 3 rejected attempts the run ends as `partial`.
- **Honest failure:** the agent may report that a goal can't be achieved. The reviewer accepts that only if the obstacle is real.
- **Post-run reflection:** a structured call extracts up to three lessons specific to the workspace and stores them for future runs.

### Traces
Every run writes `<workspace>/.sentinel/runs/<run-id>.jsonl`. It records the perceived environment, each model response (including reasoning summaries and token usage), every tool input and output, the verifier's verdicts and the reflection. It's useful for auditing what the agent did and why.

## Safety and guardrails

Guardrails limit *actions*, never reasoning (`sentinel/guardrails.py`):

- **Workspace sandbox:** file tools resolve paths and refuse anything outside the workspace. The agent can't write to its own memory or trace directory (`.sentinel/`).
- **Command blocklist:** commands such as recursive deletes of root or home, disk formatting, `sudo`, piping a download into a shell, force-push, registry edits, persistence jobs and reading SSH or cloud keys are refused in every mode. The agent is told why and picks another approach.
- **Secret scrubbing:** child processes run without environment variables that look like secrets (`*KEY*`, `*TOKEN*`, `*SECRET*`, ...), so the agent can't print your API key.
- **SSRF guard:** `fetch_url` only allows public http(s) hosts. Localhost and private-network addresses are refused.
- **Human in the loop:** `--approval risky` (the default) asks you before code runs, and `--approval all` asks before every side effect. A denied action is reported to the agent, which adapts. In a non-interactive session, anything that needs approval is denied.
- **Budgets:** hard caps on steps and spend. The agent is warned when it's close to the limit so it can wrap up cleanly.
- **Bounded outputs:** large tool outputs are trimmed (head and tail kept) so one result can't flood the context.
- **Refusal handling:** a `refusal` stop reason ends the run cleanly. By default, the API's server-side fallback (`server-side-fallback-2026-07-01` beta, `fallbacks: "default"`) re-runs a declined turn on a fallback model. Turn this off with `--no-fallback`.

## Project layout

```
sentinel/
  agent.py        the perceive → reason → act → observe → verify → reflect loop
  llm.py          Anthropic Messages API wrapper (tool turns, structured outputs, cost accounting)
  openai_compat.py  free providers (Gemini, Groq, Ollama, any OpenAI-style endpoint): message/tool
                  format translation, model auto-pick, rate-limit retries
  prompts.py      actor / verifier / reflector prompts and output schemas
  perception.py   environment snapshot and workspace change detection
  memory.py       persistent long-term memory with retrieval
  guardrails.py   sandboxing, command blocklist, approval policy, budgets
  reporting.py    live console output (rich) and the approval/ask prompts
  config.py       run configuration and model pricing
  cli.py          `sentinel run | chat | memory`
  tools/
    registry.py   tool and risk definitions, input validation
    files.py      list / read / search / write / edit
    execution.py  run_command, run_python
    web.py        fetch_url (+ server-side web_search definition)
    control.py    update_plan, remember, recall, ask_human, finish
    builtin.py    assembles the default tool set
tests/            offline tests: a scripted LLM drives the real loop
examples/         demo environments (buggy project, sales dataset)
```

## Testing

```bash
python -m pytest
```

The 43 tests run offline in under a second. They cover the guardrails, memory, every tool, perception, and the agent loop end to end, driven by a scripted stand-in for the LLM:
- the happy path through plan → act → verify → reflect
- verifier rejection sending the agent back to work
- an honest failure report
- a guardrail block reaching the model as an error
- denied approvals not running
- a re-plan after repeated failures
- a nudge after a turn with no tool call
- the step budget stopping a runaway agent
- a truncated tool call not running
- a refusal stopping the run
- memory recall across runs

## Design decisions and limits

- **Why a hand-written loop and not a framework?** The assessment is about the agent's control flow, so that flow lives in one readable file (`agent.py`). A framework would hide verification, re-plan triggers, approvals, budgets and change detection. The only runtime dependencies are `anthropic` and `rich`.
- **Providers:** the agent loop works with Anthropic-style content blocks. `openai_compat.py` translates that history to and from the OpenAI chat format, so the same loop, tools, verifier and memory run unchanged on free providers. Provider-specific extras, such as Gemini's thought signatures on tool calls, are replayed exactly as received. On Anthropic the default is `claude-opus-5-5` with adaptive thinking and `--effort` control. Smaller free models are less reliable on long multi-step goals; the verifier catches unsupported claims of success, but expect more retries.
- **Not a security boundary:** the blocklist and workspace confinement stop accidents and obvious misuse, but shell commands still run with your user's permissions. For untrusted goals, run Sentinel in a container or VM, or use `--approval all`.
- **Context:** each run is one conversation. Tool outputs are trimmed and steps are capped, which keeps a run well within the model's context window. Very long runs could add server-side compaction.
- **Memory retrieval** is keyword-based. That's enough for workspace-specific lessons, and it could be swapped for embeddings behind the same `search()` interface.

## License

MIT, see [LICENSE](LICENSE).
