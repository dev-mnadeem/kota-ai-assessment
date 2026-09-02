# ONS Annual Survey of Hours and Earnings Workplace Pensions Agent

## What this is

A working AI agent that lets a non-technical HR lead or benefits administrator at a
mid-sized UK employer ask plain-English questions about UK workplace pension trends and
get a clear, data-grounded answer back — never a raw table, never advice. It's built
against the ONS Annual Survey of Hours and Earnings (ASHE) Workplace Pensions tables
(2009–2024): P1 (age/earnings band), P2 (industry), P4 (employer size), and P5–P12
(contribution bands).

Example questions it handles:

- "What percentage of employees in the finance sector are enrolled in a defined contribution scheme?"
- "How do contribution rates for small employers compare to large ones?"
- "Has participation in construction gone up or down since 2019?"
- "Should we increase our contribution rate?" → shares the relevant data, explicitly declines to advise.

**What the system actually does, end to end:** a question comes in through the UI or API →
a LangGraph agent decides which of 7 read-only SQL tools it needs and calls them → it
writes an answer grounded only in what those tools returned → an independent LLM critic
checks that answer for groundedness, policy compliance, and clarity before it's returned
→ the answer and its sources go back to the user. Every step is LLM-driven; there is no
hardcoded intent classifier or per-question branching logic deciding what happens.

---

## How to run it

### 1. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 2. Configure your API key

Create `.env` in the repo root:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.1
OPENAI_MODEL_CANDIDATES=gpt-5.1,gpt-5,gpt-4.1,gpt-4o-mini   # optional ordered fallback
```

`data/ashe.db` is already built and checked in — no data setup is required to run the
agent (see "Data pipeline" below only if you need to rebuild it from a new ONS release).

### 3. Start the API server

```bash
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
```

Swagger UI at `http://127.0.0.1:8000/docs` lets you call `POST /api/ask` directly and see
the full response (`answer`, `sources`, `tool_calls`, `critic`) — the richest view for
understanding *why* an answer looks the way it does, not just the text.

### 4. Start the UI (new terminal)

```bash
cd ui
cp .env.example .env
npm install
npm run dev
```

Open the local URL Vite prints (usually `http://127.0.0.1:5173`) and ask a question. The
UI only shows `answer` + `sources`, by design — that's what a non-technical reader should
see.

### Prefer a script or notebook?

The agent is just a function — no CLI wrapper is needed to use it outside the API/UI:

```python
from src.agent import ask_agent
result = ask_agent("What percentage of employees in the finance sector are enrolled in a defined contribution scheme?")
print(result["answer"])
```

---

## Project structure

```
kota-assessment/
├── README.md
├── requirements.txt
├── .env                        # OPENAI_API_KEY, OPENAI_MODEL — not committed
├── .gitignore
├── data/
│   ├── ashe.db                 # the ONLY artifact the agent reads at runtime
│   ├── inventory.csv           # ETL bookkeeping: which raw files were ingested
│   └── inventory_summary.json
├── scripts/                    # one-time ETL pipeline: raw ONS files -> data/ashe.db
│   ├── inventory.py
│   ├── normalize_to_sqlite.py
│   ├── validate_db.py
│   ├── report_ingest_errors.py
│   └── ashe_data_utils.py
├── src/
│   ├── agent/                  # the LangGraph agent
│   │   ├── __init__.py         # public API: ask_agent()
│   │   ├── graph.py            # state graph: agent / tools / critic nodes + routing
│   │   ├── prompts.py          # SYSTEM_PROMPT (answering agent) + CRITIC_PROMPT (self-check)
│   │   ├── state.py            # AgentState, CriticVerdict (structured LLM output schema)
│   │   ├── tools.py            # SQLite-backed LangChain @tool functions
│   │   └── data_access_policy.py  # ACCESS_DECISIONS — retrieve/cache/hardcode as code
│   └── api/                    # thin HTTP layer over the agent
│       ├── __init__.py
│       └── server.py           # FastAPI app, POST /api/ask
├── ui/                         # Vite/React single-page front end
│   └── src/App.jsx
└── evals/                      # eval harness
    ├── golden_dataset.json
    ├── verify_golden_against_db.py
    ├── run_evals.py
    ├── llm_judge_check.py
    └── eval_report.json
```

Each `src/` package has one job: `agent/` is where all the intelligence lives (prompts,
graph, tools, structured schemas — nothing in here talks HTTP), `api/` is a thin transport
layer that does nothing but call `ask_agent()` and shape the response. `evals/` and
`scripts/` are standalone — neither imports from the other, and both can be run
independently of the API/UI.

---

## What was built and assumptions made

**A tool-calling LangGraph agent**, not a fixed decision tree. `src/agent/graph.py` is a
minimal, idiomatic 3-node loop over `MessagesState` — every decision (which tool to call,
when it has enough evidence, whether an answer needs revision) comes from the LLM, not
hardcoded Python branching:

- **`agent`** — one `ChatOpenAI` call per turn, tools bound via `.bind_tools()`. Decides
  autonomously whether to call a tool or produce a final answer.
- **`tools`** — LangGraph's prebuilt `ToolNode` executes whatever the agent requested. No
  manual dispatch table.
- **`critic`** — a structured-output LLM call that judges the latest answer on three
  independent checks (`grounded`, `advice_free`, `clear_for_hr_reader`) and can send the
  conversation back to `agent` with named issues for a bounded rewrite (max 2 attempts,
  `AGENT_MAX_CRITIC_ATTEMPTS`).

```
entry -> agent -> (tools_condition) -> tools | critic
tools -> agent
critic -> (pass / attempts exhausted) -> end
critic -> (needs revision) -> agent
```

**Both LLM roles run on detailed, explicit system prompts** (`src/agent/prompts.py`), not
short one-liners. `SYSTEM_PROMPT` spells out the agent's audience, exactly how each of the
7 tools should be used, how to represent the data's freshness/scope honestly (see
"Retrieve / cache / hardcode decisions" below — this is stated directly in the prompt, not
just enforced in code), how to handle ambiguous references and out-of-scope questions, and
the hard no-advice rule with concrete examples of disallowed language. `CRITIC_PROMPT`
defines each of its three checks with explicit pass/fail examples, refined against real
bugs found during manual testing (see "What this harness misses" below for what those
bugs were).

**Real LangChain tools**, not a hand-rolled catalog: `src/agent/tools.py` defines 7
`@tool`-decorated functions with pydantic `args_schema`s over a read-only SQLite layer
(`list_tables`, `describe_schema`, `list_dimension_values`, `lookup_industry_membership_rate`,
`industry_participation_trend`, `compare_contribution_bands_by_size`, `explain_quality_flags`).
`.bind_tools()` derives their schema automatically from the docstrings and pydantic fields —
those docstrings *are* the tool documentation the model reasons over. Every metric tool
returns a JSON dict with source tags (table/year) for grounding; when it can't resolve a
name (e.g. an unrecognized industry), it returns candidate suggestions instead of guessing,
so the LLM can self-correct.

**Two front ends**: a FastAPI server (`src/api/server.py`, `POST /api/ask`) and a small
Vite/React UI (`ui/`) that posts a question and renders the answer and sources. (An earlier
CLI wrapper was removed once the API/UI covered interactive use — `ask_agent()` remains a
plain importable function for scripting or notebook use, so nothing about how the agent
itself works changed.)

**A data pipeline**: raw ONS P-table downloads → prepared CSVs → `scripts/normalize_to_sqlite.py`
→ a single long-form `observations` table in `data/ashe.db` (245k rows, tables P1/P2/P4–P12,
2009–2024). This is the only artifact the agent reads at runtime.

### Assumptions made

- **Scope**: the runtime only ingests the `All`-segment files (not the male/female-split files ONS
  also publishes), so gender-gap questions are explicitly out of scope. The agent discovers this
  itself at query time via `describe_schema`'s `segment_values`/`work_pattern_values` fields (both
  resolve to `['All']` only) rather than a hardcoded "gender question" branch — it explains the gap
  in its own words instead of hitting a canned response.
- **"Appropriate professional guidance"** is treated as the compliance bar for advisory questions:
  acknowledge the question, retrieve and share the most relevant data, then explicitly say the
  decision needs input from a qualified adviser — never a directive recommendation, and never a bare
  deferral with no data attached.
- **Numbers only from tool calls.** The system prompt and the critic's `grounded` check both enforce
  that every fact in an answer traces back to a tool result from that conversation, not the model's
  general knowledge — this matters more than usual for LLM-generated pension numbers.
- **No conversation memory.** Every question is a fresh, single-turn call — a true follow-up like
  "and what about last year?" won't carry context from a prior question. The system prompt tells the
  agent to ask a clarifying question rather than guess when a question depends on missing context
  (e.g. "our industry" with no industry named).
- The raw ONS zip downloads and prepared CSVs used to build `data/ashe.db` are **not checked into
  this repo** (large, public, reproducible on demand) — see "Data pipeline" below for how to
  regenerate them if a new annual release needs ingesting.

---

## Retrieve / cache / hardcode decisions

The assignment asks for a deliberate, per-data-type call rather than one blanket approach. Here's
what was decided for each kind of data this agent touches, and why — answering the brief's four
questions (which data is stable enough to hardcode, the right cache strategy for annual data, what
if anything warrants live retrieval and its latency cost, and what breaks if each is gotten wrong)
per data type below. These decisions are also stated directly to the agent in `SYSTEM_PROMPT`
(`src/agent/prompts.py`), not just implemented in code — the model is explicitly told the data is a
cached annual snapshot with no live capability, so it represents that honestly in its own answers.

### 1) ASHE observations (the actual pension percentages) — **cache**

- **How:** raw ONS files → prepared CSV → normalized SQLite (`data/ashe.db`), read via SQL at query
  time.
- **Why cache, not live or hardcode:** this data updates once a year, query volume is repetitive, and
  it's far too large (245k rows) to usefully hardcode as prompt context. Local SQL is fast and
  deterministic — which matters for evals (numbers should stay stable between runs, not vary with an
  external API's uptime or a live source changing under us).
- **Latency cost:** very low at query time (local disk read, milliseconds), with an upfront one-time
  ingest step (`scripts/normalize_to_sqlite.py`) that isn't in the request path at all.
- **What breaks if this is wrong:** stale answers after a new annual ONS release, if the cache isn't
  refreshed. There's no auto-refresh — see "What I'd do next."

### 2) Taxonomy and mapping metadata (industry aliases, pension-family/employer-size labels) — **hardcode**

- **How:** a small dict in `src/agent/tools.py` (`INDUSTRY_ALIASES`) plus label-canonicalization
  helpers that normalize ONS's inconsistent year-over-year column headers (e.g. different phrasings
  of "defined contribution" across releases).
- **Why hardcode, not cache or live:** this is genuinely stable *data* — a short, reviewable lookup
  table that essentially never changes — not agent control-flow, so it's a plain dict rather than
  branching logic or a database round-trip. It's deliberately kept small: the agent's dimension-value
  resolver requires either an exact match or a *uniquely*-scored token match to auto-accept a
  mapping; on ambiguity or failure it returns candidate suggestions instead of guessing, so the LLM
  (not Python) does the disambiguating.
- **Latency cost:** near-zero — an in-memory dict lookup.
- **What breaks if this is wrong:** a tool argument resolves to the wrong dimension value and
  produces a plausible-but-wrong answer — mitigated by the uniqueness requirement and candidate
  suggestions above.

### 3) The no-advice safety policy — **LLM-driven, not hardcoded**

- **How:** `SYSTEM_PROMPT` states the rule explicitly with concrete examples of disallowed language,
  and a dedicated `critic` node (`src/agent/graph.py`, prompt in `src/agent/prompts.py`) makes a
  structured-output LLM call after every answer, judging `grounded`, `advice_free`, and
  `clear_for_hr_reader` independently, with a bounded rewrite loop if it fails.
- **Why LLM-driven, not hardcoded:** an earlier version of this agent enforced this via regex/keyword
  substring matching (`"cannot" in text and "recommend" in text`). That's brittle, doesn't generalize
  across phrasing, and puts the intelligence in Python instead of the model. Since this constraint
  requires judging *language and intent*, not just querying data, it's a decision better made by the
  LLM and checked structurally rather than by literal string matching.
- **Latency cost:** one extra LLM call per turn (the critic), plus up to one rewrite call if it fails.
- **What breaks if this is wrong:** the critic could itself misjudge an answer (false pass) — there
  is deliberately no deterministic keyword backstop underneath it, in favor of avoiding hardcoded
  logic. `evals/llm_judge_check.py` is the independent safety net that checks this boundary from
  outside the agent's own critic, using a separate, more detailed rubric.

### 4) Latest ONS release / freshness metadata — **live, but deliberately not in the request path**

- **How:** not implemented yet, but the intended shape is a periodic checker that hits ONS release
  metadata and flags when the cache needs a refresh — deliberately *not* a live call made inside a
  user's question-answering path.
- **Why live retrieval is warranted here specifically:** freshness genuinely needs an external check
  (nothing in the local cache can tell you a *newer* release exists) — this is the one place in the
  system a live call is the right tool for the job, in contrast to items 1-3 above.
- **Why NOT in the request path:** a live network call in the request path adds an avoidable external
  dependency and latency spike to every single question, for information that only changes once a
  year — the cost/benefit is completely different from a background freshness job.
- **Latency cost:** network latency and occasional upstream errors, isolated to a background/manual
  refresh job rather than every user request (versus seconds of added latency and a new failure mode
  on every question, if it were done live per-request instead).
- **What breaks if this is wrong:** stale data if freshness is never checked, or a flaky/slow user
  experience if it's checked live on every query instead.

**Operational takeaway:** the query-time path stays local and deterministic (SQLite + LLM
reasoning over it); the refresh path is explicit and separate (manual/scheduled ingest), never
hidden inside a user's question.

---

## Data pipeline

```
raw ONS P-table downloads (not in repo)
  -> prepared CSVs
  -> scripts/inventory.py       (classify + select files)
  -> scripts/normalize_to_sqlite.py  (wide -> long-form observations)
  -> data/ashe.db               (the only thing the agent reads at runtime)
```

`observations` columns: `table_id, year, segment, work_pattern, primary_group_type,
primary_group_value, earnings_band, pension_type, measure_type, value_num, value_raw,
suppression_flag, source_file`. `ingest_log` columns: `source_file, status, rows_loaded,
error_message`.

`data/ashe.db` is already built and checked in — you don't need the raw files to run the app. To
re-run ingestion from scratch (e.g. a new annual release): download the P1/P2/P4–P12 tables from
ons.gov.uk, prepare them as CSVs under a local `prepared/csv` folder, point `scripts/inventory.py`'s
`base_dir` at it, then run:

```bash
python3 scripts/inventory.py
python3 scripts/normalize_to_sqlite.py
python3 scripts/validate_db.py         # row counts, coverage by table/year
python3 scripts/report_ingest_errors.py  # summarizes any files that failed ingest
```

---

## Eval harness

Designed to separate retrieval quality from generation quality, per the assignment's ask. Verified
covering all four required elements:

| Required by the brief | Where it lives | Covered |
|---|---|---|
| A golden dataset of question and expected answer pairs | `evals/golden_dataset.json` (9 cases) | ✅ |
| A clear distinction between retrieval failures and generation failures | `evals/run_evals.py` reports these as separate counts/lists | ✅ |
| At least one LLM-as-judge check for a subjective output | `evals/llm_judge_check.py`, independent of the agent's own critic | ✅ |
| A view on what the harness would miss | "What this harness misses" below | ✅ |

| File | Purpose |
|---|---|
| `evals/golden_dataset.json` | Golden question set: expected tool calls, expected structured fields, expected answer fragments. |
| `evals/verify_golden_against_db.py` | Confirms the golden dataset's expected *numbers* still match `data/ashe.db` directly — independent of the agent entirely. |
| `evals/run_evals.py` | Runs the golden questions through the live agent and reports retrieval vs. generation failures separately. |
| `evals/llm_judge_check.py` | LLM-as-judge check for the subjective advisory-boundary case, using a detailed rubric and structured output. |
| `evals/eval_report.json` | Generated report from `run_evals.py`. |

```bash
.venv/bin/python evals/verify_golden_against_db.py
.venv/bin/python evals/run_evals.py
.venv/bin/python evals/llm_judge_check.py
```

Last verified run: `verify_golden_against_db` 6/6, `run_evals` 9/9 retrieval + 9/9 generation,
`llm_judge_check` pass with score 5/5.

### What each check catches

1. **Golden verification against source data** — fails if an expected percentage in the golden set no
   longer matches `data/ashe.db` (catches drift in the golden dataset itself, not the agent).
2. **Retrieval checks** — did the agent call the tool(s) each case expects (`expected_tool_calls`),
   does the run's aggregated source list include `expected_source`, and does the matching tool's
   *structured* JSON result contain the expected field values (float tolerance ±0.15)? For
   scope/policy cases with `expected_tool_calls: []` (gender-gap, 2030-forecast, ambiguous industry),
   it instead fails if the agent called a pension-metric tool at all — it should recognize the
   question can't be answered as asked, not fabricate a data-backed answer. There's no explicit
   "intent" field to check anymore; which tool the agent chose to call *is* the retrieval signal.
3. **Generation checks** — does the final answer text include required fragments? Counted as a
   generation failure only when retrieval already passed — i.e. the agent got the right facts but
   phrased/omitted something required.
4. **LLM-as-judge (subjective)** — a *second*, independent LLM call re-judges the advisory-boundary
   answer from scratch against a detailed rubric (three named criteria — clarity for a non-technical
   reader, use of contextualized data, and avoidance of recommendations — each with explicit
   pass/fail examples, plus a 0-5 score), returned as structured output rather than free-text. This
   check does not trust the agent's own `critic` verdict; it exists specifically so a bug in the
   critic's own judgment doesn't go unchecked by anything.

### Golden dataset design notes

- One case per core behavior: industry DC rate, industry trend, small-vs-large contribution
  comparison, the advisory boundary, plus three expected-failure-mode cases (gender-gap
  out-of-scope, an unsupported 2030 forecast, and an ambiguous reference — "our industry" with no
  industry named — where the correct behavior is a clarifying question, not a guess).
- Fragments checking **numeric facts** (`"52.9%"`) are exact substrings — the model has no reason to
  paraphrase a number. Fragments checking **policy compliance in free-form language** are
  deliberately loose (a stem like `"advi"` rather than the literal phrase `"professional
  guidance"`) — the model can satisfy the real requirement with many valid phrasings, and pinning the
  check to one exact wording produces false-fail noise on every rerun instead of catching actual
  regressions. That kind of subjective judgment is what the LLM-as-judge check is for.

### What this harness misses

- Ambiguous language handling outside the tested phrasing variants.
- Deep hallucination detection beyond the required fragments (a wrong number that happens to also
  appear elsewhere in the tool output wouldn't be caught).
- Full UI behavior and latency under load.
- Drift from upstream ONS releases, unless a refresh + re-run is performed.
- Multi-turn conversation behavior — there's no conversation memory/checkpointing yet, so every
  question is evaluated as a fresh, single-turn interaction.
- **LLM-as-judge variance.** During manual testing, the same question sent to the agent repeatedly
  occasionally produced a different critic verdict for reasons that turned out to be real bugs (a
  critic transcript that was silently truncated, and the critic's own prior rejection reasoning
  leaking back into a later judgment as if it were conversation content — both fixed) — but some
  residual inconsistency is inherent to using an LLM as a judge at all, not something a golden
  dataset of fixed questions would catch, since it shows up as run-to-run variance on the *same*
  input rather than a wrong answer to a new one.

---

## What I'd do next with more time

- **Conversation memory.** Add a `checkpointer` (LangGraph ships `MemorySaver`/thread-based
  persistence) so the agent can handle a follow-up like "and what about last year?" without repeating
  full context — currently every question is a fresh, stateless call.
- **Scheduled freshness check + refresh job**, per the "live" decision above — a small job that
  checks ONS's release calendar and flags (or triggers) a re-ingest, instead of a fully manual
  pipeline re-run.
- **Ingest the gender-segment files.** The dataset's biggest visible gap (the gender-gap question)
  is a real, answerable question ONS publishes data for — it's just not in this runtime's scope yet.
- **Tighten `describe_schema` usage for scope questions.** A question like the gender-gap one can
  cause the agent to call `describe_schema` once per table to confirm none of them have a gender
  split — correct, but slower than necessary. Worth either caching that check across a single turn
  or giving the agent a single "describe all tables" tool.
- **Broaden the golden dataset** beyond 9 cases — more industries, more employer-size pairs, more
  phrasing variants of the same underlying question.
- **Add a deeper hallucination check** to the eval harness — e.g. an LLM-as-judge pass that reads the
  full tool-call trace and the final answer and flags any claim not traceable to a tool result, not
  just the required-fragment substring check.
- **Support OR-logic in `required_answer_fragments`** (e.g. accept any of `["advi", "guidance",
  "specialist"]`) instead of a single exact-substring stem per policy-language check — would remove
  the last bit of phrasing-variance false-fail risk in the generation check without needing a full
  LLM judge for every case.
