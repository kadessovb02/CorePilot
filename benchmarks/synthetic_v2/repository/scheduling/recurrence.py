from datetime import date, datetime, timedelta


def expand_dates(start, end, weekdays=tuple(range(7))):
    if not isinstance(start, date) or isinstance(start, datetime) or not isinstance(end, date) or isinstance(end, datetime):
        raise TypeError("recurrence boundaries must be dates")
    allowed = tuple(weekdays)
    if any(isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6 for day in allowed):
        raise ValueError("weekday numbers must be integers from zero to six")
    if end < start:
        raise ValueError("end must not precede start")
    result = []
    current = start
    while current < end:
        if current.weekday() in allowed:
            result.append(current)
        current += timedelta(days=1)
    return result


def monthly_occurrences(start, months):
    if not isinstance(start, date) or isinstance(start, datetime):
        raise TypeError("start must be a date")
    if isinstance(months, bool) or not isinstance(months, int):
        raise TypeError("months must be an integer")
    if months < 0:
        raise ValueError("months must be nonnegative")
    return [start.replace(month=start.month + offset) for offset in range(months)]
