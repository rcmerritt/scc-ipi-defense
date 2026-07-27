from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
from odf.text import P

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = REPO_ROOT / "test-data"

def cell_text(cell) -> str:
    """Concatenate all <text:p> children of a TableCell into a single string."""
    parts = []
    for p in cell.getElementsByType(P):
        parts.append("".join(node.data for node in p.childNodes if node.nodeType == 3))
    return "\n".join(parts).strip()

def read_subjects(ods_path: Path) -> list[str]:
    """Return column-1 values from the first sheet, skipping a header row if present."""
    doc = load(str(ods_path))
    tables = doc.spreadsheet.getElementsByType(Table)
    if not tables:
        raise RuntimeError(f"{ods_path} has no tables")

    rows = tables[0].getElementsByType(TableRow)
    subjects: list[str] = []
    for row in rows:
        cells = row.getElementsByType(TableCell)
        if not cells:
            continue
        repeat = int(cells[0].getAttribute("numbercolumnsrepeated") or 1)
        text = cell_text(cells[0])
        # numbercolumnsrepeated on an empty cell pads the row; ignore those.
        if not text:
            continue
        subjects.append(text)

    # Drop a header row like "Subject" if present.
    if subjects and subjects[0].lower() in {"subject", "subjects"}:
        subjects = subjects[1:]

    return subjects

def numeric_key(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def build_subdir(subdir: Path) -> tuple[int, int]:
    ods = subdir / "subjects.ods"
    bodies = sorted(subdir.glob("email_*.txt"), key=numeric_key)
    if not ods.exists() or not bodies:
        return 0, 0

    subjects = read_subjects(ods)
    if len(subjects) != len(bodies):
        raise RuntimeError(
            f"{subdir.name}: {len(subjects)} subjects vs {len(bodies)} body files — counts must match"
        )

    out_path = subdir / "emails.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for body_path, subject in zip(bodies, subjects):
            record = {
                "id": body_path.stem,
                "subject": subject,
                "body": body_path.read_text(encoding="utf-8"),
                "source_file": body_path.name,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(subjects), 1

def main() -> int:
    if not TEST_DATA.is_dir():
        print(f"error: {TEST_DATA} not found", file=sys.stderr)
        return 1

    total_emails = 0
    total_dirs = 0
    for subdir in sorted(p for p in TEST_DATA.iterdir() if p.is_dir()):
        n, built = build_subdir(subdir)
        if built:
            print(f"{subdir.name}: {n} emails -> {subdir / 'emails.jsonl'}")
            total_emails += n
            total_dirs += 1
        else:
            print(f"{subdir.name}: skipped (missing subjects.ods or email_*.txt)")

    print(f"done: {total_emails} emails across {total_dirs} subdirs")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
