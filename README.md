---
tags: [agentapp, healthcare, privacy]
dataset: []
framework: []
---

# PreventNet

Federated preventive-risk review: agents share judgements, never records.

## The problem

Your pharmacy is legally forbidden from reading your GP's notes, and your lab
is legally forbidden from reading either. That's correct — GDPR purpose
limitation, and in Germany the ePA's institutional boundaries exist for good
reason. But it means a preventable harm that only shows up in the *join* of
three institutions' data — a patient on active metformin, self-reporting OTC
NSAID use, with declining kidney function, about to get a contrast CT scan —
can silently fall through the cracks. No single institution sees enough to
flag it.

PreventNet is a multi-agent preventive-medicine review built as a Flower
**AgentApp**. Three institutional agents — **GP**, **Pharmacy**, **Lab** —
each hold their own patient records that never leave their boundary. A
**Coordinator** agent asks them structured, purpose-tagged questions; they
answer only through a strict allowlisted wire format (categorical bands,
booleans — never raw records, free text, or exact values) and may **refuse**
out-of-purpose questions. Every disclosed field is written to a **disclosure
ledger**. A **Verifier** checks that every claim in the Coordinator's risk
summary is backed by a ledger entry and that the ledger contains no
cross-agent contradictions. The output is a recommendation for a **human
clinician to approve** — never a diagnosis, never a treatment.

## Related work

Existing federated-health projects mostly do **horizontal** federated
learning: the same features across many patients at many sites, training one
shared risk model (e.g. federated diagnosis-consult systems, FL risk models
across hospitals — GenoMed4All already runs Flower in EU hospital
federations). PreventNet does something different: **vertical** partitioning
of a single patient's data across institutions that legally cannot merge it,
combined with negotiated purpose-tagged disclosure (an agent can refuse a
question outright), a ledger that audits every fact an orchestrating agent
ever saw, and a deterministic verifier that catches contradictions in agent
output rather than trusting it. It also uses *vertical* federated learning,
where each institution contributes different columns for the same people, to
train a diabetes-risk model. The model's output is one more cited ledger fact,
not the decision-maker.

## Architecture

One AgentApp, one run, sequential phases. Each agent phase is an isolated
model conversation, and the federated phase keeps each department's data
inside its own object. Only wire-format ledger entries and masked, summed
federated scores reach the coordinator; raw records never do.

```
run_config: patient.id, demo.sabotage, fl.rounds, fl.lr, review.ledger
        |
        v
+-- Phase 0: Coordinator (plan) ------ model call #1
|     picks purpose-tagged questions from the closed catalog
|
+-- Phase 1: GP agent ---------------- model call #2   data/gp/records.jsonl ONLY
+-- Phase 2: Pharmacy agent ---------- model call #3   data/pharmacy/records.jsonl ONLY
+-- Phase 3: Lab agent --------------- model call #4   data/lab/records.jsonl ONLY
|     each: policy check -> answer -> strip_for_wire() -> ledger.append()
|     (sabotage flag: pharmacy answer overridden post-model, deterministically)
|
+-- Federated phase ------------------ no model calls
|     GP / Lab / Pharmacy objects each load ONLY data/<dept>/cohort.jsonl
|     vertical logistic regression, fl.rounds rounds (masked sums only)
|     -> ledger fact fl_diabetes_risk_band (low / moderate / high)
|
+-- Phase 4: Coordinator (reason) ---- model call #5 (streamed)
|     input = ONLY the ledger's wire items -> risk summary + recommendation
|
+-- Phase 5: Verifier ---------------- deterministic checks + model call #6
|     (a) every claim cites a real ledger entry
|     (b) hardcoded contradiction rules over the ledger
|     (c) allowlist re-audit of everything the coordinator saw
|     verdict: APPROVED_FOR_REVIEW | ESCALATE_CONFLICT | BLOCKED_UNVERIFIED
|
+-- Output: PREVENTNET_RESULT {json} on stdout
      + persisted to context.state ConfigRecord "preventnet"
      + human approval happens in the viewer, never auto-released
```

**Fast demo mode** (`demo.fast = true`, the default) targets a whole review in
**under 30 seconds**:
- **The departments answer with their auditable rules** (`preventnet/rules.py`), so no model call is made and no model reads records.
- **One Flower Endeavor call writes a brief assessment.** Its stream events are not forwarded token by token, because that alone slowed long runs.
- **Time budget.** If Endeavor hasn't finished within `model.budget_s` (12 s), the run switches once to `model.fallback`, and the result says so.
- **The verifier's explanation is written by code.** Federated training uses 30 rounds, which takes about 0.04 s.

Measured with an instant stub model, the app itself takes about 0.5 s including imports. The rest is SuperGrid startup plus that one model call.

Full mode (`--run-config 'demo.fast=false'`) keeps the original 6 calls: plan, three Endeavor department agents (now run in parallel), reasoning, and a verifier note. Real full-mode runs took 100–360 s, almost all of it model time; see `uv run flwr ls supergrid`.

Total model calls per run: 6 (cap enforced at `MAX_MODEL_CALLS = 8`); 1 in fast mode (2 if the fallback is used). The
federated phase adds well under a second. It prints
`PREVENTNET_FL round N/…` progress lines and emits `preventnet.fl_round` events.
If `review.ledger` is set (by the local dashboard), phases 0–3 and the federated
phase are skipped, and the run reasons over the facts it was given (2 model calls).

**How federated the SuperGrid run is.** Inside one SuperGrid task, the three
departments are separate objects. Each loads only its own folder
(`preventnet/data/<dept>/`) and gets a mask secret the coordinator code never
receives. The coordinator loop (`vfl.train_federated`) only sees masked,
summed scores and aggregate metrics. The isolation is enforced by code
structure, not by separate machines: all three departments' files ship in the
same FAB. Physically separate departments would need one Flower node per
department (the Deployment Runtime), which this project does not use. The
local deployments below run the *same* federated code as four real, separate
processes, so you can watch it.

### The core safety property

- `facet_answer(role, ...)` in `preventnet/facets.py` is the *only* function
  that opens `data/{role}/records.jsonl`, called with exactly one role per phase.
  Policy runs first, so refused questions are never shown to the model.
- `vfl.SiteModel` is the only code that opens `data/{role}/cohort.jsonl`,
  one object per department.
- The Coordinator's reasoning phase receives **only**
  `[item.as_wire_dict() for item in ledger.items]` — built from the ledger
  object, never from any variable that touched a raw record.
- `strip_for_wire()` (`preventnet/wire.py`) is an **allowlist**: it copies
  only `{field, value, band, purpose, source, ts, confidence}` and raises on
  unknown keys, oversized strings, or blocked keys (`record_id`, `text`,
  `note`, `name`, `dob`). A field added to a record later cannot leak by
  being forgotten in the allowlist.
- One question in the catalog (`q_gp_marketing_segment`) exists solely to be
  refused — every role's disclosure policy denies the `marketing` purpose, so
  the refusal path runs on every demo, not just in tests.
- A `demo.sabotage=true` run deterministically overrides the pharmacy
  agent's two safety-relevant answers *after* the model responds, then
  relies on the Verifier's hardcoded contradiction rules to catch the
  resulting inconsistency and escalate instead of recommending.

## Repository layout

```
preventnet/
  pyproject.toml
  preventnet/
    agent_app.py     AgentApp entry point, phase orchestration, events, output
    wire.py          WireItem, strip_for_wire (allowlist), Ledger
    policy.py        question catalog + per-role/purpose disclosure policy
    facets.py        facet_answer(role, ...) -> writes ledger entries/refusals
    coordinator.py   plan_questions(), reason_over_ledger()
    verifier.py      citation check, contradiction rules, verdict logic
    prompts.py       all model prompt strings
    vfl.py           vertical federated logistic regression (departments + coordinator loop)
    data/<dept>/     records.jsonl (synthetic review records) + cohort.jsonl (Pima columns), per department
  viewer/
    viewer.html      single-file, dependency-free demo UI
  scripts/
    extract_result.py   pulls PREVENTNET_RESULT from a run log -> viewer/latest.json
    prepare_data.py     splits the Pima dataset vertically into preventnet/data/<dept>/cohort.jsonl
    source_data/        pima-indians-diabetes.csv (public source; not shipped in the FAB)
  local/
    launch.py        starts the 4 local deployments
    site.py          one institutional deployment (GP / Pharmacy / Lab)
    site.html        a site's own local-only overview page
    dashboard.py     coordinator deployment: reviews, FL orchestration, approvals
    dashboard.html   clinician dashboard (embeds viewer/viewer.html)
    ui.css, icons.js shared design system for the dashboard and department pages
    rules.py         deterministic site agents
    net.py           stdlib JSON-over-HTTP helpers
  tests/
    test_invariants.py  offline safety/verifier invariants, no model calls
    test_local.py        in-process 4-deployment integration test (review, FL, sabotage, approval)
    test_ledger_mode.py  AgentApp ledger-only mode with a stubbed model client
    test_facets.py       offline facet-phase checks (stubbed model_call)
    test_coordinator.py  offline coordinator-phase checks (stubbed model_call)
```

## Demo runbook

```bash
# one-time
uv sync
uv run pytest -q
uv run flwr build
uv run flwr login supergrid

# happy path (Case A)
uv run flwr run . supergrid --stream | tee run_happy.log
python scripts/extract_result.py run_happy.log
# open viewer: cd viewer && python -m http.server 8000  → http://localhost:8000/viewer.html

# sabotage act
uv run flwr run . supergrid --run-config 'demo.sabotage=true' --stream | tee run_sabotage.log
python scripts/extract_result.py run_sabotage.log   # refresh viewer

# publish (do this at 16:00, not 17:20)
# flwr app publish flow per Flower Hub how-to; publisher in pyproject must equal your Flower username
```

At ~16:00, also record a screen capture of one successful happy-path and one
sabotage run as the fallback if SuperGrid is congested at demo time.

**3-minute script:** (0:00) "Your pharmacy is legally forbidden from reading
your GP's notes. Here's what that costs — and how agents fix it without
breaking the law." (0:20) happy path result in viewer: three boring
institutions → one caught risk. (1:10) scroll to ledger: "the coordinator's
entire world is these dozen categorical facts — and one refusal, because we
asked an out-of-purpose question on purpose." (1:50) "Now we compromise the
pharmacy agent live" → sabotage run result → amber ESCALATE, conflict
highlighted, approval disabled. (2:30) happy-path tab → clinician clicks
Approve. (2:50) Flower Hub link on screen.

## Local deployments (4 servers, vertical federated learning)

The SuperGrid AgentApp is the product. `local/` is a way to **see** it: the
same departments, policy, federated code and verifier, run as **four
independent local deployments**, each its own process and port. The
dashboard can also launch the real SuperGrid run and stream its output.

| Deployment | URL | Holds |
|---|---|---|
| GP site | http://127.0.0.1:8101 | `preventnet/data/gp/`: visit notes, diagnoses, referrals + Pima columns BMI, blood pressure, skinfold, age, pregnancies, family history **and the diabetes outcome label** |
| Pharmacy site | http://127.0.0.1:8102 | `preventnet/data/pharmacy/`: dispensing records + a synthetic `antihypertensive_dispensed` flag (Pima has no medication data) |
| Lab site | http://127.0.0.1:8103 | `preventnet/data/lab/`: eGFR/HbA1c results + Pima columns glucose, insulin |
| Coordinator dashboard | http://127.0.0.1:8100 | nothing raw: only disclosed wire items, masked/summed FL scores, aggregate metrics, approvals |

```
 GP site :8101          Pharmacy site :8102       Lab site :8103
 raw records +          raw records +             raw records +
 Pima cols + labels     synthetic col             Pima cols
 rules -> /ask          rules -> /ask             rules -> /ask
 w_gp, bias             w_ph                      w_lab
   |   \__ residuals (peer-to-peer) __/  \_______________/ |
   |                                                        |
   +---- wire items / refusals, masked partial scores ------+
                              |
                    Coordinator dashboard :8100
      ledger (re-validated) + FL risk band -> verifier -> clinician approval
                              |
          or: launch the full federated AgentApp on SuperGrid and stream its log
```

**Data.** The public [Pima Indians Diabetes dataset](https://archive.ics.uci.edu/)
comes from the US National Institute of Diabetes and Digestive and Kidney
Diseases. It covers 768 women aged 21+ of Pima Indian heritage, with no
identifiers; the source file is `scripts/source_data/pima-indians-diabetes.csv`.
- 268 of the 768 (34.9%) developed diabetes within 5 years.
- Several columns record "not measured" as 0: insulin 374 rows, skinfold 227,
  blood pressure 35, BMI 11, glucose 5.
- The dataset is split *vertically*: each department gets only the columns its
  function would hold, and the GP keeps the label.
- `uv run python scripts/prepare_data.py` regenerates
  `preventnet/data/<dept>/cohort.jsonl`, with a seeded 80/20 split (615 train /
  153 test) plus feature rows for review patients P001–P003.

The dashboard's "The data" panel shows per-column aggregates (mean, min, max,
missing) that each department computes locally.

**Where the review questions come from.** None of the ten catalog fields
(`egfr_band`, `metformin_active`, …) exist in Pima. Each department answers them
from its own synthetic review records (`records.jsonl`: visit notes, referrals,
dispensing and lab results). Every question's source and clinical rationale is
stored in `policy.QUESTION_CATALOG` (`derived_from`, `why`) and shown in the
dashboard. Pima's only role is to train the federated model, whose output
enters the ledger as `fl_diabetes_risk_band`.

**Site agents.** Each site answers catalog questions with deterministic rules
over its own records (`local/rules.py`). Flower only provides model access
inside a Flower run, and a site should not ship raw records to a remote
model, so no model sees raw data in this mode. Policy runs before any record
is read, and answers go through the same `strip_for_wire` allowlist.
Each site's own page shows its raw records, its local audit log of everything it disclosed,
and its share of the federated model. The coordinator never calls that page's API.

**Vertical federated learning** (`preventnet/vfl.py`, the same code inside the
AgentApp and here): a logistic regression for diabetes risk, trained jointly
across the three column sets. Logistic regression fits well because its
score is a plain sum (*GP part + Lab part + Pharmacy part*), so each department
computes its own part and only the parts are combined. Its weights are also
readable (each site page shows them). The dashboard only *conducts* the rounds.
1. Each site returns `X_k·w_k` plus a pairwise additive mask derived from a
   secret shared by the sites only (`PREVENTNET_FL_SECRET`, which the launcher
   gives to sites and withholds from the coordinator). Masks cancel in the sum,
   so the coordinator learns only `z = Σ X_k·w_k`. Nonces are single-use.
2. The GP (label holder) turns `z` into residuals, updates its weights, and
   sends residuals **directly** to Lab and Pharmacy. The coordinator gets only
   aggregate loss/accuracy/AUC.
3. Each site updates its own weights; weights never leave a site.
4. Entity alignment is checked blind via a digest of each site's entity IDs.

With 60 rounds this reaches ~0.79 test AUC on the held-out 153 rows. After
training, each review adds one ledger fact, `fl_diabetes_risk_band`
(low/moderate/high, source `federated-model`), computed from masked partial
scores for that patient. The dashboard learns only that one probability.

Known limits (plain VFL, no cryptography beyond masking): residuals leak label
signal to Lab/Pharmacy, and the coordinator sees per-sample summed scores.
Production VFL would add homomorphic encryption or DP noise there. The risk-band
cut points (0.15 / 0.40) are uncalibrated demo values.

**The FL chart.** The X axis is training rounds. The Y axis is a 0.4–1.0 score:
- *Train accuracy* (blue, every round): share of the 615 training people
  classified correctly at a 50% cut-off. It starts near 0.4 because an
  untrained model says 50% for everyone.
- *Test accuracy* (orange dots, every 5 rounds): the same measure on the 153
  held-out people. About 0.75.
- *Test AUC* (green dots): the probability that a random person who developed
  diabetes is scored above one who didn't (0.5 is chance, 1.0 is perfect).
  About 0.79.

Flat lines mean the model has converged, and train close to test means no
overfitting. The dashboard explains this under "How to read this chart".

**The interface.** All pages share one design system (`local/ui.css`, `local/icons.js`).
- **Dashboard** (:8100): a sidebar app. The *Overview* stays minimal: the coordinator agent banner, the three departments, the latest recommendation and an agent activity feed. *Reviews*, *Federated learning*, *Data & questions* and *Approvals* hold the detail.
- **Result page** (`viewer/viewer.html`, still a single dependency-free file, embedded in the dashboard):
  - the coordinator agent's assessment as a message card, with clickable fact citations; claims the verifier struck out are marked;
  - a verifier message;
  - the approval card and the federated diabetes-risk meter;
  - *What the coordinator saw*: the facts grouped by department, in plain words (e.g. "Kidney function (eGFR): Moderately reduced", not `egfr_band`).
- **Readable labels.** All labels come from `policy.display_labels()` and travel in the result payload.
- **Department pages** (:8101–8103): same shell, with *Overview*, *Patient records*, *Shared & declined*, *Model share* and *Cohort data*.
- **SuperGrid progress.** The raw SuperGrid log is not shown; the dashboard turns the run's `PREVENTNET_PHASE` / `PREVENTNET_FL` lines into an agent progress feed. The log only appears under "Technical details" if a run fails.

**Review engines** (the *Run on* choice on the Reviews page):
- *SuperGrid AgentApp (default)*: the dashboard runs
  `flwr run <repo> supergrid --run-config <tmp.toml> --stream` with
  `patient.id`, `demo.sabotage`, `fl.rounds` and `fl.lr`. This is the full
  federated app: Endeavor department agents, federated training inside the
  run, coordinator reasoning and the verifier. Progress appears step by step.
  When it finishes, Flower Endeavor's assessment is shown as the coordinator
  agent's message, alongside the verifier's explanation, the SuperGrid run ID
  and the training curve from *inside* the run. Requires
  `flwr login supergrid`; takes ~1–2 min.
- *SuperGrid Endeavor over local facts*: the local deployments answer with
  rules; only their disclosed facts go to SuperGrid via `review.ledger`
  (ledger-only mode, 2 model calls). The dashboard re-runs the contradiction
  rules locally and escalates if the remote verdict missed a conflict.
- *Offline*: a citation-carrying template summary, then the same verifier.
  Works with no network.

Either way the verdict is decided by code, and **Approve & send to GP** is
recorded server-side (`/api/approve`). Approval is refused (HTTP 409) for
`ESCALATE_CONFLICT` or `BLOCKED_UNVERIFIED` reviews.

### Local runbook

```bash
uv sync
uv run pytest -q
uv run python -m local.launch          # starts all four; Ctrl+C stops all
# open http://127.0.0.1:8100
#  1. Data & questions: what each department holds and where each answer comes from
#  2. Federated learning -> "Train model"  (training you can watch)
#  3. Reviews -> P001, run on "Flower SuperGrid" -> "Start review" -> watch the agents,
#     read the coordinator's assessment -> "Approve & send to GP"
#     (switch on "Simulate a compromised pharmacy agent" for the SuperGrid sabotage act)
#  4. local sabotage: Pharmacy page (http://127.0.0.1:8102) -> "Compromise the pharmacy agent",
#     run on "Offline" -> "Start review" -> "Conflict found", approval blocked
#  5. Pharmacy page -> switch it off again
```

**For a short demo:**
- **Timings.** The Reviews page shows how long each step took.
- **Fast mode.** Keep *Fast demo mode* on.
- **Pre-run.** Run P001, both normal and with the compromised pharmacy, once before presenting. *Show last result instantly* then replays those finished SuperGrid runs, clearly labelled as replays, if the venue network is slow. Replays are stored in `local/.supergrid_cache.json`, which is gitignored.
- **Training settings.** Rounds and learning rate are read when a review starts, and are locked while one runs.

`PREVENTNET_AS_OF` (default `2026-06-20`) sets the "today" the site rules use
for 90-day lookbacks on the synthetic records.

## Safety & oversight

- **Allowlist wire format**, enforced in code, not by prompt (`wire.py`).
- **Disclosure ledger**: every field an agent ever discloses is logged with
  its purpose and source; the refusal path is exercised on every run.
- **Verifier**: citation check (every risk claim must cite a real ledger
  entry, or it's struck through) + hardcoded contradiction rules (this is
  what catches a sabotaged agent) + an allowlist re-audit.
- **Human approval gate**: the recommendation is never auto-released — the
  viewer blurs it behind an explicit "Approve & send to GP" click, and an
  `ESCALATE_CONFLICT` verdict disables approval entirely, forcing the
  conflict to be resolved upstream by a human.
- **Prevention-only prompt policy**: the Coordinator is instructed to
  recommend reviews/screenings/checks only — never a diagnosis or a
  treatment change.
- **No credentials in the repo**: model access goes through
  `FLWR_RUNTIME_BASE_URL` / `FLWR_RUNTIME_API_KEY`, injected by the Flower
  Runtime. `grep -r` the repo for API keys — there are none.
- **Hard call cap**: `MAX_MODEL_CALLS = 8`; the happy path uses 6.
- **Fail loudly**: no retries/backoff beyond one bounded JSON-parse retry per
  facet call; any unhandled exception emits a `preventnet: failed` event and
  re-raises rather than silently reporting "no findings."

## Troubleshooting

- The venue's network blocked ports 9092/9093 (deployment federation
  unreachable) at a past event; everything here targets **SuperGrid over
  443**. Don't add SuperNode/deployment paths.
- The Endeavor model ref is confirmed on the day in
  `#hackathon_berlin_2026`; until then `model.ref` stays
  `"ENDEAVOR_MODEL_REF_TBD"` and the app runs on `model.fallback`
  automatically. `model_used` in the result payload is always honest about
  which model actually ran.
- SuperGrid runs execute remotely: the app never relies on writing local
  files — the `PREVENTNET_RESULT` stdout line and `context.state` are the
  only outputs.
- `publisher` in `pyproject.toml` must match your Flower account username or
  publishing will fail — fix this before 16:00.
- Strict-JSON model outputs: code fences are stripped before `json.loads`;
  one bounded retry on parse failure, then a loud failure.
- `agent.events.emit(...)` requires every event dict to include a non-empty
  string `"type"` field (undocumented in the platform facts above but
  enforced at runtime — a bare `{"preventnet": "phase", ...}` raises
  `ValueError: Run event requires a non-empty string 'type' field.`). Every
  custom event in this repo carries `"type": "preventnet.<kind>"`.
- **Dashboard waits and shows no progress.** When `flwr run` output is piped it is block-buffered, so the dashboard sets `PYTHONUNBUFFERED=1`. It stops waiting after 420 s; a run it gave up on may still finish on SuperGrid (check with `flwr ls`).
- `flwr build` only packages `.py`, `.toml`, `.md`, `.yaml`, `.yml`, `.json`,
  `.jsonl` and `LICENSE` files, so a `.csv` in `fab-include` fails with
  "did not match any files". The cohorts are therefore stored as JSONL.
- `context.locked()` does not exist on the runtime's `Context` (it's a plain
  dataclass; `state` is a `RecordDict`). Write to
  `context.state.config_records[...]` directly, no lock needed for a
  single-process AgentApp run.

## Model

`model.ref = "flower-endeavor-v1.0"` (Flower Endeavor). The literal
`"ENDEAVOR_MODEL_REF_TBD"` is kept as a sentinel: if `model.ref` is set to it,
or the first model call fails with a model-not-found error, the run switches to
`model.fallback` and emits a `preventnet.model_fallback` event. `model_used` in
the result always names the model that actually ran.

## License

MIT — see `LICENSE`.
