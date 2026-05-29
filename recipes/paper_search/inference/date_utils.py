"""Date helpers for inference-only search."""


def parse_year_month_str(value: str) -> tuple[int, int]:
    text = value.strip()
    if len(text) != 7 or text[4] != "-":
        raise ValueError(f"Expected YYYY-MM, got {value!r}")
    year, month = int(text[:4]), int(text[5:7])
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month in {value!r}")
    return year, month
