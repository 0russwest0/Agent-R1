"""Date helpers for inference-only search (Serper query bounds)."""


def parse_year_month_str(value: str) -> tuple[int, int]:
    """Parse 'YYYY-MM' into (year, month).

    Args:
        value: A string like ``2024-10``.

    Returns:
        Year and month integers.

    Raises:
        ValueError: If the string is not valid ``YYYY-MM``.
    """
    t = value.strip()
    if len(t) != 7 or t[4] != "-":
        raise ValueError(f"Expected YYYY-MM, got {value!r}")
    y, m = int(t[:4]), int(t[5:7])
    if not 1 <= m <= 12:
        raise ValueError(f"Invalid month in {value!r}")
    return y, m
