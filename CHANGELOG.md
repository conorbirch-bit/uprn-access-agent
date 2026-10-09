# Changelog

Recent application changes and the current documentation baseline. Current application: **v20.12.10**.

## v20.12.10 — Separate Saturday hours — 9 October 2026

- Add separate Saturday first-survey, last-survey-finish and return-home dropdowns, defaulting to 10:00 / 13:00 / 14:00.
- Apply the selected Saturday window to weekly planning, priority retries, weekly-note reroutes, empty-day filling and extensions of existing routes.
- Keep Monday–Friday hours unchanged and retain Saturday's retry-only restriction and existing lunch/access checks.
- Use Saturday's shorter window in capacity calculations and record its three time settings in the export.
- Verified with 140 offline regression tests, including ten Saturday-hours cases, plus syntax checks. No live routing or end-to-end Streamlit run was performed.

## v20.12.9 — Requested retry days and operational revisits — 9 October 2026

- Allocate retries with requested weekdays to available surveyors and plan those days before ordinary work, including final gap filling beyond the usual local area. Keep explicit access-only weekdays mandatory.
- Add optional Saturday availability, off by default and restricted to cannot-complete retries in every scheduling path.
- Reopen verified second/latest failed visits by Harrison Grice and the latest access-refusal failures by Joe Reynolds or Harrison Grice recorded through 9 October 2026. Assign another surveyor and retain the original failure counts. Later failures return to normal triage.
- Preserve completed-work exclusions, existing booking checks and replacement-appointment validation. Do not infer failed-visit ownership from the replacement appointment.
- Read grouped Salesforce resource names correctly and expose review evidence or missing-history limitations in the decision outputs.
- Add `Retry Day Assignments`, requested-day outcomes in `Retry Decisions`, and review evidence in the standalone access report.
- Verified with 130 offline regression tests and syntax checks. Live API routing and the Streamlit booking-selector UI tests were outside that run.

## Documentation refresh — 8 October 2026

- Replaced the accumulated v20.11.2 README with a current project overview, workflow, setup guide, input/output reference and module map.
- Documented adaptive wider filling, audited access rules, completed appointments, booking-week exclusions and the standalone client access report.
- Corrected obsolete descriptions of release-only eligibility, model segments, fixed candidate caps and the absence of final reassignment.
- Added the full-restart instructions for mixed or cached Python module versions.
- Removed report-specific examples and individual Salesforce resource identifiers from the README.
- No application code, dependencies or scheduling behaviour changed.

## v20.12.8 — Adaptive wider filling

- Retain the normal nearby/15 km filling passes, then search remaining eligible work at any distance when at least 30 minutes remain before the survey cut-off.
- Keep the survey-to-travel ratio disabled during final filling, including the wider pass.
- Preserve existing visits, duplicate protection, access rules, lunch, survey-finish and return-home deadlines.
- Allow empty-day fallback beyond the initial three working areas when that shortlist cannot produce a route.
- Add the `Day Filling Review` worksheet and a `Fill Pass` field in `Gap Filling`.
- Revise `Capacity Review` so survey hours alone do not imply a shortage of work; travel also consumes the working day.
- Record the wider-search trigger and limits in `Run Settings`.
- Verified with 99 offline regression tests, including ten new wider-search cases, plus syntax and app/export integration checks. Live API routing and the Streamlit booking-selector UI tests were outside that run.

## v20.12.7 — Final filling without an efficiency ratio

- Remove the minimum survey-to-travel ratio from final filling while retaining the 15 km search radius.
- Retain initial planning's efficiency preference and the original day-priority and empty-day area selection.
- Keep operational constraints and access reporting in place.

## v20.12.5 — Standalone access report

- Add separate generation and download of the cannot-completes report without building a weekly schedule.
- Include `Summary`, `Client Access Required` and `All Cannot Completes` tabs.
- Preserve original surveyor descriptions and date-only visit history, with blank client response fields and a response-status selector.
- Keep client-help cases independent of selected booking-exclusion weeks and remove resolved Work Orders from client help.
- Include mapping-only completed records in the register, leaving unavailable source details blank.

## Earlier foundations retained in the current app

- Access rules informed by a manual audit of approximately 100 buildings: separate customer and operational failures, interpret blank reasons as no answer, and escalate clear access barriers before the two-customer-failure limit.
- Completed Service Appointment matching and narrowly scoped approvals for previously resolved access issues.
- Current-week detection and selectable booking-week exclusions based on unstarted replacement appointments.
- Final team filling from the eligible portfolio, with workload and unplaced-building diagnostics.
- Separate garage, 1–3-flat, 4–6-flat and 7+-flat duration models, including labelled missing-input fallbacks.
- Improved missing-coordinate handling, candidate-batch exhaustion and workload allocation for surveyors with smaller initial allocations.
- Explicit routing service errors and retained completed AI triage decisions when a later batch fails.

For the current behaviour and installation steps, see [README.md](README.md) and [RUN_THIS_VERSION.txt](RUN_THIS_VERSION.txt).
