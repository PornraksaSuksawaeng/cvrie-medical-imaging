import csv
from pathlib import Path

import pandas as pd

COLUMNS = ("id", "color", "testimony")


# NOTE: The raw file is malformed.
def read_testimonies(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV into a tidy frame with one row per testimony.

    The file has no header and a variable number of fields per line:
    ``id,color,testimony`` when the colour is present, ``id,testimony`` when
    it is not. Feeding it to ``pd.read_csv`` with three column names fills
    left to right, which silently moves the testimony into the colour column
    for every short row, so we parse it by hand instead.
    """
    records = []

    with open(path, newline="", encoding="utf-8-sig") as fh:
        for line, fields in enumerate(csv.reader(fh), start=1):
            if not fields or not any(field.strip() for field in fields):
                continue
            if len(fields) == 3:
                patient_id, color, testimony = fields
            elif len(fields) == 2:
                patient_id, testimony = fields
                color = None
            else:
                raise ValueError(
                    f"{path}:{line}: expected 2 or 3 fields, got {len(fields)}"
                )
            records.append((int(patient_id), color, testimony))

    frame = pd.DataFrame.from_records(records, columns=COLUMNS)
    frame["color_present"] = frame["color"].notna()
    return frame
