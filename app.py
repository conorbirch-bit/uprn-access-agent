"""Streamlit interface for the UPRN Building Access Agent."""

from __future__ import annotations

import hashlib
from pathlib import Path

import csv
from datetime import datetime

import streamlit as st

from note_storage import (
    load_site_notes,
    save_site_note,
)

from Voice_notes import transcribe_audio

from agent import (
    CONTACT,
    CONTACT_CONFIRMED,
    CONTACT_EMAIL,
    DROP_SWITCH,
    EXTERNAL_ACCESS,
    EXTERNAL_DETAILS,
    EXTERNAL_KEY,
    INTERNAL_CODE_REQUIRED,
    INTERNAL_DESCRIPTION,
    INTERNAL_KEY_REQUIRED,
    INTERNAL_KEY_TYPE,
    NO_RANGE,
    NOTES,
    OTHER_CONTACT_DETAILS,
    PRIMARY_PHONE,
    SECONDARY_PHONE,
    UPRNAccessAgent,
    format_address,
    validate_columns,
)

from xlsx_reader import read_records


DEFAULT_WORKBOOK = Path(__file__).with_name("Access_Information.xlsx")

SITE_NOTES_FILE = Path(__file__).with_name("site_notes.csv")

st.set_page_config(
    page_title="UPRN Building Access Agent",
    page_icon="🏢",
    layout="centered",
)


@st.cache_resource(show_spinner="Reading access information…")
def load_agent(
    source_key: str,
    file_bytes: bytes | None,
) -> UPRNAccessAgent:

    source = (
        file_bytes
        if file_bytes is not None
        else DEFAULT_WORKBOOK
    )

    records = read_records(
        source,
        sheet_name="sheet1",
    )

    missing = validate_columns(records)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    return UPRNAccessAgent(records)


def yes_no(value: str) -> str:
    return value if value else "Not recorded"

def save_site_note(uprn: str, note: str) -> None:
    file_exists = SITE_NOTES_FILE.exists()

    with SITE_NOTES_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "UPRN",
                    "Date Time",
                    "Site Access Note",
                ]
            )

        writer.writerow(
            [
                uprn.strip().upper(),
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                note.strip(),
            ]
        )
def load_site_notes(uprn: str) -> list[dict]:
    if not SITE_NOTES_FILE.exists():
        return []

    with SITE_NOTES_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        notes = [
            row
            for row in reader
            if row["UPRN"].strip().upper()
            == uprn.strip().upper()
        ]

    # Newest notes first
    return list(reversed(notes))

# -----------------------------
# PAGE HEADER
# -----------------------------

st.title("UPRN Building Access Agent")

st.write(
    "Enter a UPRN to receive a conversational summary "
    "of the building address, access arrangements, "
    "contact details and access risk."
)


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("Data source")

    uploaded = st.file_uploader(
        "Use a different Excel workbook",
        type=["xlsx"],
    )

    st.caption(
        "The included workbook is used automatically. "
        "Uploading another file does not overwrite it."
    )

    st.divider()

    st.markdown(
        "**High risk rule**  \n"
        "Internal controlled access is required **and** "
        "no backup option is recorded."
    )


# -----------------------------
# LOAD WORKBOOK
# -----------------------------

try:

    if uploaded is not None:

        workbook_bytes = uploaded.getvalue()

        workbook_hash = hashlib.sha256(
            workbook_bytes
        ).hexdigest()

        agent = load_agent(
            f"upload:{workbook_hash}",
            workbook_bytes,
        )

    else:

        if not DEFAULT_WORKBOOK.exists():

            st.error(
                "The default workbook could not be found. "
                "Upload an .xlsx file."
            )

            st.stop()

        workbook_bytes = DEFAULT_WORKBOOK.read_bytes()

        workbook_hash = hashlib.sha256(
            workbook_bytes
        ).hexdigest()

        agent = load_agent(
            f"default:{workbook_hash}",
            workbook_bytes,
        )

except Exception as exc:

    st.error(
        f"The workbook could not be loaded: {exc}"
    )

    st.stop()


# -----------------------------
# UPRN SEARCH
# -----------------------------

uprn = st.text_input(
    "UPRN",
    placeholder="For example: ACTO0023",
    help="The lookup ignores spaces and letter case.",
)

search_clicked = st.button(
    "Find building",
    type="primary",
    use_container_width=True,
)


# -----------------------------
# LOOKUP RESULT
# -----------------------------

if search_clicked or uprn:

    if not uprn.strip():

        st.warning("Enter a UPRN first.")

    else:

        result = agent.lookup(uprn)

        if result is None:

            st.error(
                f"No building was found for UPRN "
                f"‘{uprn.strip()}’."
            )

            suggestions = agent.suggestions(uprn)

            if suggestions:

                st.write(
                    "Closest recorded UPRNs: "
                    + ", ".join(suggestions)
                )

        else:

            record = result.record
            risk = result.risk


            # -----------------------------
            # ACCESS RISK
            # -----------------------------

            if risk.level == "High":

                st.error(
                    f"Access risk: {risk.level}"
                )

            elif risk.level == "Medium":

                st.warning(
                    f"Access risk: {risk.level}"
                )

            else:

                st.success(
                    f"Access risk: {risk.level}"
                )


            # -----------------------------
            # SUMMARY
            # -----------------------------

            st.markdown(
                "### Conversational summary"
            )

            st.write(result.summary)


            # -----------------------------
            # BUILDING
            # -----------------------------

            st.markdown("### Building")

            st.write(
                f"**Address:** "
                f"{format_address(record)}"
            )

            if record.get(NO_RANGE):

                st.write(
                    f"**Property range:** "
                    f"{record.get(NO_RANGE)}"
                )


            # -----------------------------
            # ACCESS
            # -----------------------------

            st.markdown(
                "### Access details"
            )

            external_col, internal_col = st.columns(2)

            with external_col:

                st.markdown("**External**")

                st.write(
                    "**Trade button/intercom:** "
                    f"{yes_no(record.get(EXTERNAL_ACCESS, ''))}"
                )

                st.write(
                    "**Times/other information:** "
                    f"{yes_no(record.get(EXTERNAL_DETAILS, ''))}"
                )

                st.write(
                    "**Fire-brigade drop switch:** "
                    f"{yes_no(record.get(DROP_SWITCH, ''))}"
                )

                st.write(
                    "**External key type:** "
                    f"{yes_no(record.get(EXTERNAL_KEY, ''))}"
                )


            with internal_col:

                st.markdown("**Internal**")

                st.write(
                    "**Fob/key required:** "
                    f"{yes_no(record.get(INTERNAL_KEY_REQUIRED, ''))}"
                )

                st.write(
                    "**Key/fob type:** "
                    f"{yes_no(record.get(INTERNAL_KEY_TYPE, ''))}"
                )

                st.write(
                    "**Key code required:** "
                    f"{yes_no(record.get(INTERNAL_CODE_REQUIRED, ''))}"
                )

                st.write(
                    "**Description:** "
                    f"{yes_no(record.get(INTERNAL_DESCRIPTION, ''))}"
                )


            # -----------------------------
            # CONTACT DETAILS
            # -----------------------------

            st.markdown("### Contact")

            contact_col, number_col = st.columns(2)

            with contact_col:

                st.write(
                    "**Name:** "
                    f"{yes_no(record.get(CONTACT, ''))}"
                )

                st.write(
                    "**Confirmed:** "
                    f"{yes_no(record.get(CONTACT_CONFIRMED, ''))}"
                )

                st.write(
                    "**Email:** "
                    f"{yes_no(record.get(CONTACT_EMAIL, ''))}"
                )


            with number_col:

                st.write(
                    "**Primary number:** "
                    f"{yes_no(record.get(PRIMARY_PHONE, ''))}"
                )

                st.write(
                    "**Secondary number:** "
                    f"{yes_no(record.get(SECONDARY_PHONE, ''))}"
                )

                st.write(
                    "**Other possible contact details:** "
                    f"{yes_no(record.get(OTHER_CONTACT_DETAILS, ''))}"
                )


            # -----------------------------
            # VOICE NOTE
            # -----------------------------

            st.divider()

            st.subheader(
                "🎤 Site access note"
            )

            st.caption(
                "Record any access issues encountered "
                "at this building."
            )

            uprn_key = uprn.strip().upper()

            note_key = (
                f"note_box_{uprn_key}"
            )

            if note_key not in st.session_state:
                st.session_state[note_key] = ""


            audio = st.audio_input(
                "Record access note",
                key=f"audio_{uprn_key}",
            )


            if audio is not None:

                st.audio(audio)

                if st.button(
                    "Transcribe note",
                    key=f"transcribe_{uprn_key}",
                ):

                    try:

                        with st.spinner(
                            "Transcribing voice note..."
                        ):

                            transcript = transcribe_audio(
                                audio,
                                st.secrets[
                                    "OPENAI_API_KEY"
                                ],
                            )

                        # Temporary debugging line.
                        # This tells us what OpenAI returned.
                        st.write(
                            "Transcript received:",
                            repr(transcript),
                        )

                        st.session_state[
                            note_key
                        ] = transcript

                    except Exception as exc:

                        st.error(
                            "Transcription failed: "
                            f"{exc}"
                        )


            note = st.text_area(
                "Review or edit note",
                key=note_key,
                height=140,
            )

        if st.button(
            "💾 Save site note",
            key=f"save_note_{uprn_key}",
        ):
            if not note.strip():
                st.warning(
                "There is no note to save."
                )
            else:
                try:
                    save_site_note(
                    uprn_key,
                    note,
                )

                    st.success(
                    "Site access note saved."
                )

                except Exception as exc:
                    st.error(
                    f"Could not save note: {exc}"
                )


            st.markdown("### Previous site access notes")
            
            previous_notes = load_site_notes(uprn_key)
            
            if not previous_notes:
                st.caption(
                    "No previous site access notes have been saved."
                )
            
            else:
                for saved_note in previous_notes:
                    st.markdown(
                        f"**{saved_note['Date Time']}**"
                    )
            
                    st.write(
                        saved_note["Site Access Note"]
                    )
            
                    st.divider()
            
                    
            
                        # -----------------------------
                        # EXISTING NOTES
                        # -----------------------------
            
                    if record.get(NOTES):
            
                        st.info(
                            f"Notes: "
                            f"{record.get(NOTES)}"
                            )
            

            # -----------------------------
            # DEBUG SOURCE DATA
            # -----------------------------

            with st.expander(
                "Show source row"
            ):

                st.json(
                    dict(record)
                )
