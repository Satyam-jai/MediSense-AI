"""
mathutils.py
------------
A small user-defined module used to demonstrate Python's module/import
mechanism (Unit 1). It provides simple statistical helpers used later by
the risk-scoring layer (Unit 4) to summarize a patient's vitals.

Import mechanism demonstrated:
    from app.utils.mathutils import mean, clamp, normalize
or
    import app.utils.mathutils as mathutils
"""

from typing import Iterable


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean of a sequence of numbers."""
    values = list(values)
    if not values:
        raise ValueError("mean() requires at least one value")
    return sum(values) / len(values)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to the inclusive range [low, high]."""
    return max(low, min(value, high))


def normalize(value: float, low: float, high: float) -> float:
    """
    Scale `value` from the range [low, high] to [0, 1].
    Used to normalize vitals (e.g., heart rate, BP) before feeding
    them into ML models in later units.
    """
    if high == low:
        raise ValueError("normalize() requires high != low")
    value = clamp(value, low, high)
    return (value - low) / (high - low)


# __name__ guard: this block only runs when the module is executed directly
# (e.g. `python mathutils.py`), not when it is imported elsewhere.
if __name__ == "__main__":
    sample = [72, 80, 95, 110, 65]
    print("Sample vitals:", sample)
    print("Mean:", mean(sample))
    print("Clamped 130 to [60,120]:", clamp(130, 60, 120))
    print("Normalized 95 in [60,120]:", round(normalize(95, 60, 120), 3))
