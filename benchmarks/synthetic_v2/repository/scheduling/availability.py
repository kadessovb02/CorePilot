from datetime import timedelta

from .time_math import Window


def merge_windows(windows):
    rows = sorted(windows, key=lambda window: window.start)
    result = []
    for window in rows:
        if result and window.start < result[-1].end:
            previous = result[-1]
            result[-1] = Window(previous.start, max(previous.end, window.end))
        else:
            result.append(window)
    return result


def find_slot(windows, duration, not_before=None):
    if not isinstance(duration, timedelta):
        raise TypeError("duration must be a timedelta")
    if duration <= timedelta(0):
        raise ValueError("duration must be positive")
    for window in sorted(windows, key=lambda value: value.start):
        candidate = max(window.start, not_before) if not_before is not None else window.start
        if candidate + duration < window.end:
            return candidate
    return None


def remaining_capacity(resource, bookings, capacity):
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an integer")
    if capacity < 0:
        raise ValueError("capacity must be nonnegative")
    active = sum(not booking.cancelled for booking in bookings)
    return max(0, capacity - active)
