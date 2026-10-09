# UPRN Building Access Agent

A local Streamlit application that reads the supplied Excel workbook and lets a user search by UPRN. It returns:

- the building address and property range;
- external and internal access arrangements;
- contact details;
- a conversational summary; and
- an access risk rating with an explanation.

No external AI service or API key is required. The summary is generated locally from the spreadsheet fields, so contact and access data stays on the machine running the app.

## Risk logic

The user-defined **High** risk condition is implemented exactly as a combined rule:

1. Internal controlled access is required, based on either:
   - `Internal - is a fob or key required? = Yes`; or
   - `Internal - is a key code required? = Yes`.
2. No backup option is recorded.

A backup option is currently recognised when at least one of these is present:

- trade button or intercom access is available;
- a fire-brigade drop switch is available;
- the external access notes explicitly describe an alternative route; or
- a secondary contact number is recorded.

The backup definition is contained in `_backup_options()` in `agent.py`, making it easy to adjust after the business confirms exactly what should count as a backup.

Other ratings are included to make the output more useful:

- **Medium:** only one of the two high-risk conditions applies.
- **Low:** controlled internal access is not required and a backup is recorded.

## Run the web app on Windows

Open PowerShell in this folder, then run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The application opens in a browser. Type a UPRN and select **Find building**.

## Use on a phone on the same Wi-Fi network

Run:

```powershell
streamlit run app.py --server.address 0.0.0.0
```

Open the network address displayed by Streamlit on the phone. The computer must remain running.

## Command-line alternative

```powershell
python cli.py ACTO0023
```

## Workbook requirements

The included `Access_Information.xlsx` uses the existing column names. A replacement workbook can be uploaded through the sidebar, provided it contains the same headers.
