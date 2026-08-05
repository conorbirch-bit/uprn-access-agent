"""Minimal XLSX reader using only the Python standard library.

This keeps the agent dependency-light and reads values from the first worksheet
without changing the source workbook.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping
import re
import xml.etree.ElementTree as ET
import zipfile

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

Source = str | Path | bytes | BinaryIO


def _open_source(source: Source):
    if isinstance(source, bytes):
        return BytesIO(source)
    return source


def _column_number(reference: str) -> int:
    match = re.match(r"([A-Z]+)", reference.upper())
    if not match:
        raise ValueError(f"Invalid cell reference: {reference}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - 64
    return result


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    strings: list[str] = []
    with archive.open("xl/sharedStrings.xml") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag == f"{{{MAIN_NS}}}si":
                text = "".join(
                    node.text or "" for node in element.iter(f"{{{MAIN_NS}}}t")
                )
                strings.append(text)
                element.clear()
    return strings


def _sheet_targets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))

    relationship_map = {
        relation.attrib["Id"]: relation.attrib["Target"]
        for relation in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
    }

    sheets_node = workbook.find(f"{{{MAIN_NS}}}sheets")
    if sheets_node is None:
        return []

    targets: list[tuple[str, str]] = []
    for sheet in sheets_node:
        name = sheet.attrib["name"]
        relationship_id = sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
        target = relationship_map[relationship_id]
        if not target.startswith("xl/"):
            target = "xl/" + target.lstrip("/")
        targets.append((name, target))
    return targets


def _cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        inline = cell.find(f"{{{MAIN_NS}}}is")
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.iter(f"{{{MAIN_NS}}}t"))

    value_node = cell.find(f"{{{MAIN_NS}}}v")
    if value_node is None:
        return ""

    raw_value = value_node.text or ""
    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)]
        except (ValueError, IndexError):
            return raw_value
    if cell_type == "b":
        return "Yes" if raw_value == "1" else "No"
    return raw_value


def iter_rows(source: Source, sheet_name: str | None = None) -> Iterator[list[str]]:
    """Yield worksheet rows as text values."""
    with zipfile.ZipFile(_open_source(source)) as archive:
        shared_strings = _load_shared_strings(archive)
        sheets = _sheet_targets(archive)
        if not sheets:
            raise ValueError("No worksheets were found in the workbook.")

        if sheet_name is None:
            _, target = sheets[0]
        else:
            matches = [target for name, target in sheets if name == sheet_name]
            if not matches:
                available = ", ".join(name for name, _ in sheets)
                raise ValueError(
                    f"Worksheet '{sheet_name}' was not found. Available: {available}"
                )
            target = matches[0]

        with archive.open(target) as stream:
            for _, row_element in ET.iterparse(stream, events=("end",)):
                if row_element.tag != f"{{{MAIN_NS}}}row":
                    continue

                values: dict[int, str] = {}
                for cell in row_element.findall(f"{{{MAIN_NS}}}c"):
                    reference = cell.attrib.get("r", "A1")
                    values[_column_number(reference)] = _cell_text(cell, shared_strings)

                if values:
                    last_column = max(values)
                    yield [values.get(index, "") for index in range(1, last_column + 1)]
                row_element.clear()


def read_records(source: Source, sheet_name: str | None = None) -> list[dict[str, str]]:
    """Read the first non-empty row as headers and return record dictionaries."""
    row_iterator = iter_rows(source, sheet_name)

    headers: list[str] | None = None
    for row in row_iterator:
        if any(str(value).strip() for value in row):
            headers = [str(value).strip() for value in row]
            break

    if headers is None:
        return []

    records: list[dict[str, str]] = []
    for row in row_iterator:
        padded = row + [""] * max(0, len(headers) - len(row))
        record = {
            header: str(padded[index]).strip()
            for index, header in enumerate(headers)
            if header
        }
        if any(record.values()):
            records.append(record)
    return records
