"""Independent scheduling reference and deliberately incomplete changes for CPU audits."""
import ast
from pathlib import Path


MUTATIONS = {
    "fix_duration_ceiling": ("scheduling/time_math.py", "duration_minutes", '''def duration_minutes(start, end):
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        raise TypeError("boundaries must be datetimes")
    if end < start:
        raise ValueError("end must not precede start")
    elapsed = end - start
    micros = ((elapsed.days * 86400 + elapsed.seconds) * 1000000) + elapsed.microseconds
    return (micros + 59999999) // 60000000
'''),
    "fix_half_open_overlap": ("scheduling/time_math.py", "overlaps", '''def overlaps(left, right):
    return left.start < right.end and right.start < left.end
'''),
    "fix_empty_intersection": ("scheduling/time_math.py", "clip_window", '''def clip_window(window, horizon):
    start, end = max(window.start, horizon.start), min(window.end, horizon.end)
    return Window(start, end) if start < end else None
'''),
    "fix_recurrence_end": ("scheduling/recurrence.py", "expand_dates", '''def expand_dates(start, end, weekdays=tuple(range(7))):
    if not isinstance(start, date) or isinstance(start, datetime) or not isinstance(end, date) or isinstance(end, datetime):
        raise TypeError("recurrence boundaries must be dates")
    allowed = tuple(weekdays)
    if any(isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6 for day in allowed):
        raise ValueError("weekday numbers must be integers from zero to six")
    if end < start:
        raise ValueError("end must not precede start")
    result = []
    current = start
    while True:
        if current.weekday() in allowed:
            result.append(current)
        if current == end:
            return result
        current += timedelta(days=1)
'''),
    "fix_month_rollover": ("scheduling/recurrence.py", "monthly_occurrences", '''def monthly_occurrences(start, months):
    import calendar
    if not isinstance(start, date) or isinstance(start, datetime):
        raise TypeError("start must be a date")
    if isinstance(months, bool) or not isinstance(months, int):
        raise TypeError("months must be an integer")
    if months < 0:
        raise ValueError("months must be nonnegative")
    result = []
    for offset in range(months):
        year, zero_month = divmod(start.year * 12 + start.month - 1 + offset, 12)
        month = zero_month + 1
        result.append(date(year, month, min(start.day, calendar.monthrange(year, month)[1])))
    return result
'''),
    "fix_adjacent_merge": ("scheduling/availability.py", "merge_windows", '''def merge_windows(windows):
    rows = sorted(windows, key=lambda window: window.start)
    result = []
    for window in rows:
        if result and window.start <= result[-1].end:
            previous = result[-1]
            result[-1] = Window(previous.start, max(previous.end, window.end))
        else:
            result.append(window)
    return result
'''),
    "fix_exact_fit_slot": ("scheduling/availability.py", "find_slot", '''def find_slot(windows, duration, not_before=None):
    if not isinstance(duration, timedelta):
        raise TypeError("duration must be a timedelta")
    if duration <= timedelta(0):
        raise ValueError("duration must be positive")
    for window in sorted(windows, key=lambda value: value.start):
        candidate = max(window.start, not_before) if not_before is not None else window.start
        if candidate + duration <= window.end:
            return candidate
    return None
'''),
    "fix_resource_capacity": ("scheduling/availability.py", "remaining_capacity", '''def remaining_capacity(resource, bookings, capacity):
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an integer")
    if capacity < 0:
        raise ValueError("capacity must be nonnegative")
    active = sum(booking.resource == resource and not booking.cancelled for booking in bookings)
    return max(0, capacity - active)
'''),
    "fix_utc_conversion": ("scheduling/reminders.py", "utc_timestamp", '''def utc_timestamp(when):
    if not isinstance(when, datetime):
        raise TypeError("when must be a datetime")
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("when must be timezone-aware")
    return when.astimezone(timezone.utc).isoformat(timespec="seconds")
'''),
    "fix_reminder_direction": ("scheduling/reminders.py", "reminder_time", '''def reminder_time(start, lead=timedelta(minutes=15)):
    if not isinstance(start, datetime) or not isinstance(lead, timedelta):
        raise TypeError("start and lead must be datetime and timedelta")
    if lead < timedelta(0):
        raise ValueError("lead must be nonnegative")
    return start - lead
'''),
    "fix_single_cancellation": ("scheduling/service.py", "drop_booking", '''def drop_booking(bookings, identifier):
    return [booking for booking in bookings if booking.identifier != identifier]
'''),
    "fix_chronological_agenda": ("scheduling/service.py", "sorted_agenda", '''def sorted_agenda(bookings):
    return sorted(bookings, key=lambda booking: booking.start)
'''),
}


COUNTEREXAMPLES = {
    "fix_duration_ceiling": ("micros + 59999999", "micros + 30000000"),
    "fix_half_open_overlap": ("left.start < right.end and right.start < left.end", "left.start < right.end"),
    "fix_empty_intersection": ("if start < end else None", "if start <= end else None"),
    "fix_recurrence_end": ("allowed = tuple(weekdays)", "allowed = tuple(range(7))"),
    "fix_month_rollover": ("min(start.day, calendar.monthrange(year, month)[1])", "min(28, start.day)"),
    "fix_adjacent_merge": ("max(previous.end, window.end)", "window.end"),
    "fix_exact_fit_slot": ("candidate = max(window.start, not_before) if not_before is not None else window.start", "candidate = window.start"),
    "fix_resource_capacity": ("booking.resource == resource and not booking.cancelled", "booking.resource == resource"),
    "fix_utc_conversion": ("when.astimezone(timezone.utc)", "when.replace(tzinfo=timezone.utc)"),
    "fix_reminder_direction": ("return start - lead", "return start - timedelta(minutes=15)"),
    "fix_single_cancellation": ("booking.identifier != identifier", "booking.identifier != identifier and booking.resource != 'room-a'"),
    "fix_chronological_agenda": ("key=lambda booking: booking.start", "key=lambda booking: (booking.start, booking.identifier)"),
}


def replace_function(workspace, case_id, replacement):
    relative, name, _ = MUTATIONS[case_id]
    path = Path(workspace) / relative
    original = path.read_text()
    node = next(node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef) and node.name == name)
    lines = original.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [replacement]
    source = "".join(lines)
    ast.parse(source)
    path.write_text(source)


def apply_reference(workspace, case_id):
    replace_function(workspace, case_id, MUTATIONS[case_id][2])


def apply_counterexample(workspace, case_id):
    old, new = COUNTEREXAMPLES[case_id]
    reference = MUTATIONS[case_id][2]
    if old not in reference:
        raise ValueError("counterexample no longer matches")
    replace_function(workspace, case_id, reference.replace(old, new))
