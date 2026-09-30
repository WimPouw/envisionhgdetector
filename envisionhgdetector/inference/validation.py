"""Validate numeric inference options."""

def valid_float(value: float) -> bool:
    return value >= 0 and value <= 1.0

