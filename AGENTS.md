# ProbeBench research execution

The user requested sustained autonomous development with parallel subagents, local
compute only, reproducible experiments and working release artifacts. Keep the full
four-study programme in PLAN.md; a prototype is not completion of the research goal.
Website redesign is assigned to Claude. Read STATUS.md, current Git state and the next
incomplete plan item before work. Preserve unrelated changes and completed evidence.

Use matching gstack planning, review, testing and shipping skills. One agent owns
integration; delegate bounded tasks with non-overlapping writes. Independent review
should inspect evidence, not trust summaries. If agent capacity is exhausted, record
it and continue appropriate local work rather than repeatedly spawning failing agents.

The integration task has a native persisted goal and an hourly continuation. Inspect
their actual state when needed; don't claim an agent is running from an old status file.
Do not create duplicate schedulers or spend reset credits without explicit permission.
Subagents should report concrete artifacts and checks to the integration owner.

Scientific correctness outranks apparent progress. Distinguish OPD from mirror surface
height, compute WFE on the sampled pupil, preserve camera noise and fixed image scaling,
and account for all exposures and expected incident photons. Policies must never receive
hidden truth, private seeds or evaluator metrics. Current in-process isolation is an
API boundary, not a sandbox; do not run untrusted policy code inside it.

Keep development, selection and final evaluation separate. Never redefine the existing
controller study's frozen protocol to fit local resource limits. Existing noether-ao work
and its scheduler are outside this checkout. Do not overwrite its dirty files or use its
reserved cohorts for debugging. Record new scientific assumptions in DECISIONS.md.

Use the pinned environment from a source checkout:

```sh
uv sync --locked --extra dev
uv run --locked python -m pytest -q
uv run --locked python -m probebench.run --output work/unique-development-run
uv run --locked python -m probebench.run --verify work/unique-development-run
```

Tests must catch physical, information-boundary, accounting or evidence-integrity
regressions. Retain every trial and numerical failure. Never infer speed from a simulated
loop rate, asynchronous dispatch or a correctness-suite runtime. Review raw evidence and
independent checks before making a scientific claim. Cite only opened sources.

Recommend Astra for unresolved physics, contracts and interpretation; Sol for settled
implementation and execution; Luna only for mechanical edits. Never claim settings changed
unless an available control confirms it. Local compute only excludes paid APIs/cloud
purchases. No public upload, visibility/licence change or messaging people is authorized.
Private repository development is authorized. New branches use the `codex/` prefix.

Each completed increment updates STATUS.md with actual checks, retained run paths,
limitations and one next action. Stop adding work when the full goal is complete or an
unchanged external blocker leaves no useful authorized action. Use native goal status
according to its real completion/blocked criteria, not as a substitute for evidence.
