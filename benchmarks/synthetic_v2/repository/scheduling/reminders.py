from datetime import datetime, timedelta, timezone


def utc_timestamp(when):
    if not isinstance(when, datetime):
        raise TypeError("when must be a datetime")
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("when must be timezone-aware")
    return when.replace(tzinfo=timezone.utc).isoformat(timespec="seconds")


def reminder_time(start, lead=timedelta(minutes=15)):
    if not isinstance(start, datetime) or not isinstance(lead, timedelta):
        raise TypeError("start and lead must be datetime and timedelta")
    if lead < timedelta(0):
        raise ValueError("lead must be nonnegative")
    return start + lead
