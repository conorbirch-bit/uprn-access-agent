# Site Survey Scheduling Agent

An AI-assisted planning application that turns a building portfolio and survey history into weekly schedules for a team of surveyors. It combines duration prediction, public-transport routing and operational rules, then exports schedules and access reports to Excel.

**Current application: v20.12.10** · [Recent changes](CHANGELOG.md) · [Run and update notes](RUN_THIS_VERSION.txt)

Developed by **Conor Birch** during an operational improvement project at Metro Safety covering approximately 1,600 buildings. The project's reported operational impact was an approximately **50% increase in time spent on billable work**.

## What it does

- Predicts survey duration from completed surveys and building characteristics.
- Allocates work across surveyors with different home locations and working days.
- Builds routes using Google public-transport journey times and configurable time limits.
- Adds remaining eligible work to short days, expanding beyond the usual 15 km search when necessary.
- Distinguishes routine failed-visit retries from cases requiring client help or internal review.
- Produces a Salesforce upload worksheet, scheduling diagnostics and a separate cannot-completes access report.

## How the planning works

1. **Import and predict.** Normalise Excel/Salesforce reports and estimate survey durations using scikit-learn Ridge regression. Model families cover garages, 1–3 flats, 4–6 flats and 7+ flats, with fallbacks for missing inputs.
2. **Apply operational rules.** Check drawing/date readiness, completed appointments, existing bookings and failed-visit decisions before adding work to the scheduling pool.
3. **Group and allocate.** Use coordinates and postcodes to identify working areas, then allocate candidates using surveyor availability, workload and home-to-area journey estimates. Optional AI advice considers current work and future nearby demand.
4. **Build daily routes.** Evaluate public-transport journeys against the survey window, lunch and return-home deadline. Configurable local-transfer assumptions avoid unnecessary Google calls for nearby buildings.
5. **Fill remaining time.** Share globally unbooked eligible work across the team, including buildings outside the original shortlists. Try nearby work first, then widen the search for underfilled days.
6. **Export and review.** Download the team schedule, Salesforce copy, unplaced work and decision records. Generate the client access report from the same retry rules.

Language models support cluster selection, supported location requests, unfamiliar access descriptions and the final narrative. Python rules and route checks determine whether a visit can be scheduled. Updated training data can be uploaded to refit the duration model; it does not collect new outcomes automatically.

## Current scheduling rules

The app plans **one selected week at a time**, using `Europe/London` dates. Each surveyor has individually selected working days and a start/finish location. The planner enforces the first-survey target, latest survey finish, latest return home and configured buffers. A 30-minute lunch starts between 11:45 and 13:00 on days extending into the lunch period.

### Requested retry days and Saturday

Retries with suggested weekdays are assigned to surveyors available on those days and planned before ordinary work. Final team filling also tries the requested days first, including candidates outside the usual 15 km area. A suggestion can fall back to another feasible day when the requested day cannot be used; instructions such as **only available Friday** remain hard constraints. Explicit requests take precedence over the generic different-weekday rule for no-answer retries.

Saturday is available as an optional checkbox for each surveyor and is off by default. **Only cannot-complete retries can be scheduled on Saturday**, including during final filling. Fresh buildings remain restricted to the selected weekdays.

Saturday has separate dropdowns for **first survey start**, **last survey finish** and **latest return home**, defaulting to **10:00 / 13:00 / 14:00**. These adjustable times apply to initial planning, requested-day priority, weekly-note reroutes and final filling. Monday–Friday use the original time controls. Capacity calculations use the shorter Saturday window, and all three Saturday settings appear in `Run Settings`. The existing lunch and travel checks still apply.

All portfolio Work Types and Statuses are considered. Eligibility still depends on usable location and duration data, drawing/date readiness and access checks. A linked **Completed Service Appointment** resolves its Work Order in the retry workflow; a portfolio label such as **Work Done** is not the same completion signal.

### Fuller days in v20.12.8

Normal filling tries nearby work, including candidates within 15 km. If **at least 30 minutes remain before the survey cut-off**, a final pass considers remaining eligible buildings without a distance cap. Final filling has **no minimum survey-to-travel ratio**; initial route planning retains its efficiency preference.

The wider pass appends visits while preserving already placed visits and all hard constraints. It can increase routing requests and run time. It cannot guarantee a full day when no feasible additional visit fits, and it does not globally reorder existing routes.

### Failed visits and client access

The access rules were refined through a manual audit of approximately **100 buildings**, using the surveyor descriptions and recorded outcomes.

| Situation | Current treatment |
| --- | --- |
| A linked Service Appointment is Completed | Resolve the Work Order and exclude it from retry/client-help work. |
| Two or more recorded customer failures | Require client help unless the history qualifies for an approved resolution or the operational revisit review below. |
| Clear access barrier after a failed visit | Request client help even below the two-customer-failure threshold; examples include refused access, an unusable entry system or a required appointment/escort. |
| Blank failure description | Assess as “No answer at door”; retain the original source text unchanged in the report. |
| Routine no-answer below the customer limit | Retry on a different weekday unless an explicit requested day says otherwise, with a preference for the opposite morning/afternoon period. |
| Operational failure attributed to Metro | Track separately from customer failures; a routine operational retry has no automatic weekday ban. |
| Known access days or unresolved technical/instruction issues | Enforce the access days, or hold the work for internal review. |
| Approved resolved issue | Reopen only the reviewed failure history; a fresh failure goes through the normal checks. |

Unrecognised descriptions can use AI triage. Cases without a valid decision remain on hold. Scheduling a retry also requires a usable, unambiguous replacement Service Appointment.

The **9 October 2026 operational review** reopens unresolved work whose second and latest failed visit was by **Harrison Grice**, plus the latest access-refusal failures recorded by **Joe Reynolds or Harrison Grice** on or before that date. These visits go to a different surveyor. Original customer/Metro counts are retained; a new failure after the review returns to normal triage. Completed appointments, selected booking exclusions and replacement-appointment checks still apply.

The review uses names and dates linked to the actual failed Service Appointment, not the owner of a replacement appointment. Include the failed-visit `Resource Name` in the export. Salesforce grouped names are carried down within explicitly grouped columns; missing names, dates or incomplete visit history are reported without guessing the review exception. `Retry Decisions` and the access report's `All Cannot Completes` tab expose the review evidence.

### Avoiding repeat bookings

The **Weeks to exclude existing bookings** selector defaults to the current UK week and accepts multiple weeks. It reads `Scheduled Start` from the appointment-mapping tab to exclude unstarted replacement appointments in those weeks.

A known failed visit is treated separately from an outstanding booking, so a fresh unscheduled replacement can still pass through the normal retry checks. Refresh the Salesforce report after booking a week before planning another one: the application uses uploaded data rather than a live Salesforce connection. The booking filter applies to the retry Work Orders represented in that report; it is not a general live-calendar duplicate check.

## Excel outputs

The weekly workbook includes:

| Worksheet | Purpose |
| --- | --- |
| Team Summary / Full Team Schedule | Team totals and scheduled visits, lunch and return journeys. |
| Salesforce Copy | The configured 10-column Field Service import layout, excluding lunch/return rows. |
| Capacity Review / Unplaced Eligible | Available workload and eligible buildings left out of the schedule. |
| Day Filling Review / Gap Filling | Wider-search decisions and additional visits; Gap Filling appears when work was added. |
| Retry Day Assignments | Candidate assignments to surveyors available on requested weekdays; final placement still depends on route feasibility. |
| Booking Exclusions / Retry Decisions / Client Access Required | Booking and access decisions when a retry workbook is supplied. |
| Run Settings | Application version, selected week, time windows and wider-search settings. |

Additional sheets record clusters, allocations, source portfolio, drawing priority and individual surveyor schedules.

`Retry Decisions` includes **Requested Day Honoured?** and an outcome explanation, so an assignment is not confused with an actual booking.

The separate **cannot-completes access report** contains `Summary`, `Client Access Required` and `All Cannot Completes`. It includes original surveyor descriptions, recorded failed-visit dates in date-only format, failure counts and suggested client actions. Client-help cases are independent of booking-week exclusions. Blank response/contact/access-date fields support client follow-up; responses from a previous export are not imported automatically.

## Run locally

Use **Python 3.10 or newer**; recent offline checks used Python 3.12. From the project folder, preferably in a virtual environment:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Provide your completed-surveys training workbook as `Predictive Model.xlsx` beside `app.py`, or upload it through the sidebar. The app trains the duration model during startup, including when you only intend to generate an access report.

Set credentials through environment variables or `.streamlit/secrets.toml`:

```toml
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_ROUTES_KEY"
OPENAI_API_KEY = "YOUR_OPENAI_KEY"
OPENAI_MODEL = "YOUR_ENABLED_MODEL"

# Optional disruption/weather context
TFL_API_KEY = ""
MET_OFFICE_API_KEY = ""
MET_OFFICE_GLOBAL_SPOT_URL = ""
```

Google routing is required to generate schedules. OpenAI is required for enabled AI features. To plan without it, disable **AI cluster selection** and **Generate one AI summary of the final team week**, and leave weekly notes blank. Unfamiliar access cases requiring AI will remain on hold without a valid AI decision. TfL and Met Office provide optional context for the final narrative.

Keep credentials and operational workbooks outside the published repository. Team defaults, Salesforce resource mappings and approved access exceptions are deployment-specific and should be reviewed when adapting the project to another organisation.

### Input workbooks

Use `.xlsx` files. The importers support ordinary tables and the Salesforce report layouts handled by the project, including title/filter rows above the table.

| Input | Main fields |
| --- | --- |
| Completed-surveys training | `Building Height`, `Internal Ground Floor Area (m2)`, `Sovereign Flat`, `Primary Service Appointment: Actual Duration (Minutes)`. These columns must exist; missing values use the supported model fallbacks where possible. |
| Master portfolio | Building/reference identifiers, `Postcode`, building characteristics, drawing/date readiness and the Work Order/Service Appointment IDs needed for Salesforce export. Latitude/longitude improve geographic grouping. |
| Optional cannot-completes workbook | Two tabs: failure history with reasons, customer/Metro counts and failed-visit `Resource Name`; appointment mapping with Work Order number, Service Appointment ID, appointment status, `Actual Start`, `Scheduled Start` and location data. |

In **Weekly scheduling**, upload the relevant files, select the week and available surveyors, set the time windows and booking exclusions, then generate and review the schedule. To produce just the access report, upload the retry workbook and select **Generate cannot-completes report**.

## Project structure

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit interface, workflow coordination and weekly exports. |
| `duration_predictor_height.py` | Segmented duration models and prediction diagnostics. |
| `coordinate_clustering.py` / `portfolio_clusterer.py` | Geographic grouping, readiness and portfolio planning. |
| `team_scheduler.py` | Team allocation, workload repair and final gap filling. |
| `scheduler_v20_10.py` | Active daily/weekly route engine and feasibility checks. |
| `google_routes.py` | Google transit routes and route-matrix requests. |
| `ai_planner.py` / `special_requests.py` | AI planning advice and constrained weekly location requests. |
| `retry_planner.py` / `access_rules.py` | Appointment matching, booking exclusions and access decisions. |
| `access_report.py` | Standalone cannot-completes Excel report. |
| `salesforce_master.py` | Salesforce/standard portfolio import. |
| `tfl_client.py` / `metoffice_client.py` | Optional disruption and weather context. |
| `tests/` | Routing, retry, booking, report and filling regression cases. |

The application version and module filename differ: **v20.12.10 still imports `scheduler_v20_10.py`**. Older scheduler modules are retained in the project but are not the app's active route engine.

## Validation and limits

With the dependencies installed, run:

```bash
python -m unittest discover -s tests -v
```

The v20.12.10 change passed **140 offline regression tests**, including ten separate-Saturday-hours cases alongside requested-day priority, reviewed revisits and Saturday restrictions, plus syntax checks. That run excluded the Streamlit booking-selector UI tests; it did not exercise live Google/OpenAI calls or an end-to-end Streamlit schedule.

The planner uses heuristics rather than a proof of the best possible team schedule. Predictions depend on the supplied survey history; missing inputs and small training segments reduce reliability. Nearby transfers can use configured local assumptions rather than measured journeys. Short days can remain because of transit, return-home limits, access days or insufficient feasible work. Use the diagnostic worksheets to distinguish these cases.

After updating Python files, fully stop and restart Streamlit. A browser refresh can leave an older imported module in memory; see [run and update notes](RUN_THIS_VERSION.txt) for the `expand_underfilled_days` version-mismatch fix.
