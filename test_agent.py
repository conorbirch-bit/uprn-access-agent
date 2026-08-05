from agent import (
    DROP_SWITCH,
    EXTERNAL_ACCESS,
    INTERNAL_CODE_REQUIRED,
    INTERNAL_KEY_REQUIRED,
    SECONDARY_PHONE,
    UPRN,
    UPRNAccessAgent,
    assess_risk,
)


def test_high_risk():
    record = {
        UPRN: "TEST0001",
        INTERNAL_KEY_REQUIRED: "Yes",
        INTERNAL_CODE_REQUIRED: "No",
        EXTERNAL_ACCESS: "No",
        DROP_SWITCH: "No",
        SECONDARY_PHONE: "",
    }
    assert assess_risk(record).level == "High"


def test_not_high_with_backup():
    record = {
        UPRN: "TEST0002",
        INTERNAL_KEY_REQUIRED: "Yes",
        INTERNAL_CODE_REQUIRED: "No",
        EXTERNAL_ACCESS: "Yes",
        DROP_SWITCH: "No",
        SECONDARY_PHONE: "",
    }
    assert assess_risk(record).level == "Medium"


def test_lookup_ignores_case_and_spaces():
    agent = UPRNAccessAgent([{UPRN: "ABCD0001"}])
    assert agent.lookup(" abcd0001 ") is not None
