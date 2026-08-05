"""Streamlit interface for the UPRN Building Access Agent."""

from __future__ import annotations
import hashlib
from pathlib import Path
import streamlit as st

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

st.set_page_config(
    page_title="UPRN Building Access Agent",
    page_icon="🏢",
    layout="centered",
)


@st.cache_resource(show_spinner="Reading access information…")
def load_agent(source_key: str, file_bytes: bytes | None) -> UPRNAccessAgent:
    source = file_bytes if file_bytes is not None else DEFAULT_WORKBOOK
    records = read_records(source)
    missing = validate_columns(records)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    return UPRNAccessAgent(records)


def yes_no(value: str) -> str:
    return value if value else "Not recorded"


st.title("UPRN Building Access Agent")
st.write(
    "Enter a UPRN to receive a conversational summary of the building address, "
    "access arrangements, contact details and access risk."
)

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Use a different Excel workbook", type=["xlsx"])
    st.caption(
        "The included workbook is used automatically. Uploading another file does "
        "not overwrite it."
    )
    st.divider()
    st.markdown(
        "**High risk rule**  \n"
        "Internal controlled access is required **and** no backup option is recorded."
    )

try:
    if uploaded is not None:
        workbook_bytes = uploaded.getvalue()
        agent = load_agent(f"upload:{uploaded.name}:{len(workbook_bytes)}", workbook_bytes)
    else:
    if not DEFAULT_WORKBOOK.exists():
        st.error("The default workbook could not be found. Upload an .xlsx file.")
        st.stop()

    workbook_bytes = DEFAULT_WORKBOOK.read_bytes()
    workbook_hash = hashlib.sha256(workbook_bytes).hexdigest()

    agent = load_agent(
        f"default:{workbook_hash}",
        workbook_bytes,
    )
except Exception as exc:
    st.error(f"The workbook could not be loaded: {exc}")
    st.stop()

uprn = st.text_input(
    "UPRN",
    placeholder="For example: ACTO0023",
    help="The lookup ignores spaces and letter case.",
)
search_clicked = st.button("Find building", type="primary", use_container_width=True)

if search_clicked or uprn:
    if not uprn.strip():
        st.warning("Enter a UPRN first.")
    else:
        result = agent.lookup(uprn)
        if result is None:
            st.error(f"No building was found for UPRN ‘{uprn.strip()}’.")
            suggestions = agent.suggestions(uprn)
            if suggestions:
                st.write("Closest recorded UPRNs: " + ", ".join(suggestions))
        else:
            record = result.record
            risk = result.risk

            if risk.level == "High":
                st.error(f"Access risk: {risk.level}")
            elif risk.level == "Medium":
                st.warning(f"Access risk: {risk.level}")
            else:
                st.success(f"Access risk: {risk.level}")

            st.markdown("### Conversational summary")
            st.write(result.summary)

            st.markdown("### Building")
            st.write(f"**Address:** {format_address(record)}")
            if record.get(NO_RANGE):
                st.write(f"**Property range:** {record.get(NO_RANGE)}")

            st.markdown("### Access details")
            external_col, internal_col = st.columns(2)
            with external_col:
                st.markdown("**External**")
                st.write(f"Trade button/intercom: {yes_no(record.get(EXTERNAL_ACCESS, ''))}")
                st.write(f"Times/other information: {yes_no(record.get(EXTERNAL_DETAILS, ''))}")
                st.write(f"Fire-brigade drop switch: {yes_no(record.get(DROP_SWITCH, ''))}")
                st.write(f"External key type: {yes_no(record.get(EXTERNAL_KEY, ''))}")
            with internal_col:
                st.markdown("**Internal**")
                st.write(f"Fob/key required: {yes_no(record.get(INTERNAL_KEY_REQUIRED, ''))}")
                st.write(f"Key/fob type: {yes_no(record.get(INTERNAL_KEY_TYPE, ''))}")
                st.write(f"Key code required: {yes_no(record.get(INTERNAL_CODE_REQUIRED, ''))}")
                st.write(f"Description: {yes_no(record.get(INTERNAL_DESCRIPTION, ''))}")

            st.markdown("### Contact")
            contact_col, number_col = st.columns(2)
            with contact_col:
                st.write(f"**Name:** {yes_no(record.get(CONTACT, ''))}")
                st.write(f"**Confirmed:** {yes_no(record.get(CONTACT_CONFIRMED, ''))}")
                st.write(f"**Email:** {yes_no(record.get(CONTACT_EMAIL, ''))}")
                st.write(f"**Other possible contact details:** "f"{yes_no(record.get(OTHER_CONTACT_DETAILS, ''))}"
    )
            with number_col:
                st.write(f"**Primary number:** {yes_no(record.get(PRIMARY_PHONE, ''))}")
                st.write(f"**Secondary number:** {yes_no(record.get(SECONDARY_PHONE, ''))}")

            st.markdown("### Risk reasoning")
            st.write(risk.explanation)
            st.write(
                "**Internal controlled access required:** "
                + ("Yes" if risk.requires_internal_access else "No")
            )
            st.write(
                "**Backup options recorded:** "
                + (", ".join(risk.backup_options) if risk.backup_options else "None")
            )

            if record.get(NOTES):
                st.info(f"Notes: {record.get(NOTES)}")

            with st.expander("Show source row"):
                st.json(dict(record))
