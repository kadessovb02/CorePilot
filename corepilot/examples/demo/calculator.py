"""Synthetic demo for exercising a coding agent without real project data."""
from decimal import Decimal


def total(prices: list[str], discount_percent: int = 0) -> Decimal:
    if not 0 <= discount_percent <= 100:
        raise ValueError("discount_percent must be between 0 and 100")
    values = [Decimal(price) for price in prices]
    if any(value < 0 for value in values):
        raise ValueError("prices cannot be negative")
    return (sum(values, Decimal(0)) * (100 - discount_percent) / 100).quantize(Decimal("0.01"))
