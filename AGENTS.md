# AGENTS.md

## Backlog Management

- Distinguish between:
  - surface work vs runtime work,
  - non-recursive support vs recursive support,
  - targeted/local execution vs full optimizer or rewrite semantics.
- Close completed slices as soon as their shipped boundary is real, tested, and documented. Do not leave them hanging under one parent item "for later."
- If one narrow residual blocker remains, promote that residual blocker into its own explicit item instead of keeping the whole broader item open.
- When a backlog item has accumulated more than 2-3 landed slices, reassess whether it should be split before continuing further work.
- Backlog wording must track the real shipped contract:
  - completed behavior belongs under closed/completed slices,
  - only genuinely unshipped behavior stays under the open item,
  - avoid status text that makes completed work look perpetually partial.
- Do not silently defer known work; record it in `docs/plans/` with concrete next steps.

## Mandatory Session Report Rule

- Append a dated entry to `docs/SESSION_REPORT.md` only when the session produces meaningful reportable information. Avoid low-signal incremental updates that fragment the report.
- Prefer updating the existing entry for the same date/session over appending a new same-day entry. Create multiple entries on one date only when there is a genuinely separate, decision-relevant milestone with new runtime results or conclusions.
- Create a new dated entry when at least one of the following is true:
  1. Significant technical conclusions have been reached.
  2. An experiment has concluded with reportable results or decision-relevant metrics.
  3. The user explicitly directs the agent to end, wrap up, or record the session.
- Do not create a report entry for routine environment setup, dependency installation, lightweight inspections, or documentation-only passes unless they materially changed the technical recommendation or produced a decision-relevant failure/surprise.
- When an entry is warranted, include:
  - objectives attempted
  - code/config changes made
  - experiment commands and key metrics
  - best current checkpoint/config recommendation
  - unresolved issues and next actions
  - a final `Signature:` line with a stable agent identifier so entries from different agents can be distinguished later
- If no code changed but a meaningful conclusion or decision was reached, add a short inspection-only entry. Do not add placeholder entries when the session produced no meaningful new information.
- Signature requirements:
  - put the signature at the end of the entry
  - use a stable identifier for that agent across sessions, for example `Signature: Codex (GPT-5)` or `Signature: Gemini CLI Agent`
  - if an agent name/model is unknown, use the most specific stable handle available rather than omitting the signature
- A session is not complete until any required report update under this rule is saved.

## Forward-Risk Directive

- When the active line of experimentation is clearly stalling or consuming long wall-clock time on minor safe tweaks, move forward with sensible architectural or optimization risk instead of defaulting to the lowest-risk local change.
- Required behavior:
  1. Prefer changes that can materially change the regime over changes that only slightly smooth or calibrate the current regime.
  2. Do not spend multiple sessions on tiny “safe” tweaks once the pattern of failure is already clear.
  3. When choosing the next step, bias toward the smallest change that can still falsify the current bottleneck hypothesis decisively.
  4. State clearly when a live run must be restarted for a code change to matter; do not leave an old process running and talk as if the new code is active.

## Runtime Launch Guardrails

- Do not start or continue a long-running training job on CPU when the machine has an available supported GPU and the model/workload is large enough that GPU execution is the expected regime.
- For this repository, value training, state-transition training, and state-policy training must default to GPU execution on GPU-capable hosts.
  - long runs should set `WAVEPDE_DEVICE=gpu` and treat missing/failed GPU initialization as a hard stop, not a CPU fallback.
- Before launching any long-running training job expected to last more than a few minutes, explicitly verify:
  1. whether a supported GPU is present,
  2. whether the current codepath actually moves model parameters and batches onto that device,
  3. whether a short smoke run confirms real device utilization.
- If GPU support is not wired yet, stop and wire it before launching the long run. Do not use a long CPU run as a placeholder benchmark on a GPU-capable machine unless the user explicitly approves that tradeoff.
- If observability is incomplete, fix logging/checkpoint visibility before launching the long run. Do not leave a high-cost process running if the first optimization step and progress path cannot be verified from logs.
