import csv
from pathlib import Path

FRACTIONS = (0.6, 0.8)


def cut(length, fraction):
    return max(1, min(length - 1, round(length * fraction)))


def example_id(record, fraction):
    return f"{record.family}_{record.sequence_id}_{int(fraction * 100)}"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
