"""UPRN access lookup and risk assessment logic."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from typing import Iterable, Mapping
import re

UPRN = "UPRN"
PROPERTY_ID = "Property ID"
NO_RANGE = "No Range"
BUILDING_NAME = "Building Name"
STREET_NO = "Street No"
STREET = "Street"
TOWN = "Town"
POSTCODE = "Pcode"

EXTERNAL_ACCESS = "External - can you access via a trade button or intercom?"
EXTERNAL_DETAILS = "External - what times are they available or any other information?"
DROP_SWITCH = "External - is a fire brigade drop switch available?"
EXTERNAL_KEY = "External - what type of key"

INTERNAL_KEY_REQUIRED = "Internal - is a fob or key required?"
INTERNAL_KEY_TYPE = "Internal - what type of key"
INTERNAL_DESCRIPTION = "Internal - Add a description if needed"
INTERNAL_CODE_REQUIRED = "Internal - is a key code required?"

CONTACT = "Contact"
CONTACT_CONFIRMED = "Contact Confirmed"
CONTACT_EMAIL = "Contact email"
PRIMARY_PHONE = "Primary Contact Number"
SECONDARY_PHONE = "Secondary Contact Number"
OTHER_CONTACT_DETAILS = "other possible Contact Details"
NOTES = "Notes"

REQUIRED_COLUMNS = {
    UPRN,
    BUILDING_NAME,
    STREET_NO,
    STREET,
    TOWN,
    POSTCODE,
    EXTERNAL_ACCESS,
    EXTERNAL_DETAILS,
    DROP_SWITCH,
    EXTERNAL_KEY,
    INTERNAL_KEY_REQUIRED,
    INTERNAL_KEY_TYPE,
    INTERNAL_DESCRIPTION,
    INTERNAL_CODE_REQUIRED,
    CONTACT,
    CONTACT_EMAIL,
    PRIMARY_PHONE,
    SECONDARY_PHONE,
    NOTES,
}

YES_VALUES = {"yes", "y", "true", "1", "required", "available"}
ALTERNATIVE_ACCESS_WORDS = {
    "also",
    "alternative",
    "backup",
    "concierge",
    "caretaker",
    "key safe",
    "keysafe",
    "key fob",
    "fob access",
}


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalise_uprn(value: object) -> str:
    return re.sub(r"\s+", "", clean(value)).upper()


def is_yes(value: object) -> bool:
    return clean(value).lower() in YES_VALUES


def present(value: object) -> bool:
    return bool(clean(value))


@dataclass(frozen=True)
class RiskAssessment:
    level: str
    requires_internal_access: bool
    backup_options: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class LookupResult:
    record: Mapping[str, str]
    risk: RiskAssessment
    summary: str


def validate_columns(records: list[Mapping[str, str]]) -> list[str]:
    if not records:
        return sorted(REQUIRED_COLUMNS)
    existing = set(records[0].keys())
    return sorted(REQUIRED_COLUMNS - existing)


def build_index(records: Iterable[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    index: dict[str, Mapping[str, str]] = {}
    for record in records:
        key = normalise_uprn(record.get(UPRN, ""))
        if key:
            index[key] = record
    return index


def _requires_internal_access(record: Mapping[str, str]) -> bool:
    return is_yes(record.get(INTERNAL_KEY_REQUIRED)) or is_yes(
        record.get(INTERNAL_CODE_REQUIRED)
    )


def _backup_options(record: Mapping[str, str]) -> tuple[str, ...]:
    """Return explicitly recorded alternative ways to gain or arrange access.

    The rule is intentionally centralised so it can be changed easily if the
    organisation adopts a different definition of a backup option.
    """
    options: list[str] = []

    if is_yes(record.get(EXTERNAL_ACCESS)):
        options.append("trade button or intercom")
    if is_yes(record.get(DROP_SWITCH)):
        options.append("fire-brigade drop switch")

    details = clean(record.get(EXTERNAL_DETAILS)).lower()
    if details and any(word in details for word in ALTERNATIVE_ACCESS_WORDS):
        options.append("alternative external access noted")

    # A secondary contact does not open the building directly, but it provides
    # a fallback route for arranging access when the primary contact fails.
    if present(record.get(SECONDARY_PHONE)):
        options.append("secondary contact")

    return tuple(dict.fromkeys(options))


def assess_risk(record: Mapping[str, str]) -> RiskAssessment:
    internal_access = _requires_internal_access(record)
    backups = _backup_options(record)

    # User-defined high-risk rule: BOTH conditions must be true.
    if internal_access and not backups:
        level = "High"
        explanation = (
            "Internal doors require controlled access and no backup access option "
            "is recorded. Access should be confirmed before attending."
        )
    elif internal_access or not backups:
        level = "Medium"
        if internal_access:
            explanation = (
                "Internal controlled access is required, but at least one backup "
                "option is recorded."
            )
        else:
            explanation = (
                "No internal controlled-access requirement is recorded, but there "
                "is no clear backup option."
            )
    else:
        level = "Low"
        explanation = (
            "No internal controlled-access requirement is recorded and at least "
            "one backup option is available."
        )

    return RiskAssessment(level, internal_access, backups, explanation)


def format_address(record: Mapping[str, str]) -> str:
    first_line_parts = [
        clean(record.get(BUILDING_NAME)),
        clean(record.get(STREET_NO)),
        clean(record.get(STREET)),
    ]
    first_line = " ".join(part for part in first_line_parts if part)
    remaining = [clean(record.get(TOWN)), clean(record.get(POSTCODE))]
    return ", ".join(part for part in [first_line, *remaining] if part) or "Not recorded"


def _sentence(label: str, value: object, fallback: str = "Not recorded") -> str:
    text = clean(value)
    return f"{label}: {text or fallback}."


def create_summary(record: Mapping[str, str], risk: RiskAssessment) -> str:
    uprn = clean(record.get(UPRN))
    address = format_address(record)
    unit_range = clean(record.get(NO_RANGE))

    external_bits = []
    if is_yes(record.get(EXTERNAL_ACCESS)):
        external_bits.append("a trade button or intercom is recorded as available")
    else:
        external_bits.append("no trade button or intercom is recorded as available")

    if is_yes(record.get(DROP_SWITCH)):
        key_type = clean(record.get(EXTERNAL_KEY))
        external_bits.append(
            "a fire-brigade drop switch is available"
            + (f" using {key_type}" if key_type else "")
        )
    else:
        external_bits.append("no fire-brigade drop switch is recorded")

    external_details = clean(record.get(EXTERNAL_DETAILS))
    if external_details:
        external_bits.append(external_details)

    internal_bits = []
    if is_yes(record.get(INTERNAL_KEY_REQUIRED)):
        key_type = clean(record.get(INTERNAL_KEY_TYPE))
        internal_bits.append(
            "an internal fob or key is required"
            + (f" ({key_type})" if key_type else "")
        )
    else:
        internal_bits.append("no internal fob or key requirement is recorded")

    if is_yes(record.get(INTERNAL_CODE_REQUIRED)):
        internal_bits.append("an internal key code is also required")

    internal_description = clean(record.get(INTERNAL_DESCRIPTION))
    if internal_description:
        internal_bits.append(internal_description)

    contact_parts = []

    if present(record.get(CONTACT)):
        contact_parts.append(clean(record.get(CONTACT)))

    if present(record.get(CONTACT_EMAIL)):
        contact_parts.append(clean(record.get(CONTACT_EMAIL)))

    if present(record.get(PRIMARY_PHONE)):
        contact_parts.append(clean(record.get(PRIMARY_PHONE)))

    if present(record.get(SECONDARY_PHONE)):
        contact_parts.append(
            f"backup: {clean(record.get(SECONDARY_PHONE))}"
        )

    if present(record.get(OTHER_CONTACT_DETAILS)):
        contact_parts.append(
            f"other possible contact details: "
            f"{clean(record.get(OTHER_CONTACT_DETAILS))}"
        )

    contact_text = (
        ", ".join(contact_parts)
        if contact_parts
        else "No contact details are recorded"
    )

    notes = clean(record.get(NOTES))
    unit_text = f" The recorded property range is {unit_range}." if unit_range else ""
    note_text = f" Additional note: {notes}" if notes else ""

    return (
        f"UPRN {uprn} is at {address}.{unit_text} "
        f"For external entry, {'; '.join(external_bits)}. "
        f"For internal access, {'; '.join(internal_bits)}. "
        f"The recorded contact details are {contact_text}. "
        f"Access risk is {risk.level.lower()}: {risk.explanation}{note_text}"
    )


class UPRNAccessAgent:
    def __init__(self, records: list[Mapping[str, str]]):
        self.records = records
        self.index = build_index(records)

    def lookup(self, uprn: str) -> LookupResult | None:
        record = self.index.get(normalise_uprn(uprn))
        if record is None:
            return None
        risk = assess_risk(record)
        return LookupResult(record=record, risk=risk, summary=create_summary(record, risk))

    def suggestions(self, uprn: str, limit: int = 5) -> list[str]:
        return get_close_matches(
            normalise_uprn(uprn), list(self.index.keys()), n=limit, cutoff=0.45
        )
