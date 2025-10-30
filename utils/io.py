from pathlib import Path
import csv

def ensure_csv(path: str, header: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with p.open('w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerow(header)

def append_csv(path: str, row: list) -> None:
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        csv.writer(fh).writerow(row)
