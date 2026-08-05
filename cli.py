"""Optional command-line version of the UPRN Building Access Agent."""

from pathlib import Path
import sys

from agent import UPRNAccessAgent
from xlsx_reader import read_records

WORKBOOK = Path(__file__).with_name("Access_Information.xlsx")


def main() -> int:
    uprn = sys.argv[1] if len(sys.argv) > 1 else input("Enter UPRN: ")
    agent = UPRNAccessAgent(read_records(WORKBOOK))
    result = agent.lookup(uprn)
    if result is None:
        print(f"No building found for UPRN: {uprn}")
        suggestions = agent.suggestions(uprn)
        if suggestions:
            print("Closest matches:", ", ".join(suggestions))
        return 1
    print(result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
