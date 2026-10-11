from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    def __post_init__(self):
        if not isinstance(self.start, datetime) or not isinstance(self.end, datetime):
            raise TypeError("window boundaries must be datetimes")
        if self.end <= self.start:
            raise ValueError("window must have positive duration")


@dataclass(frozen=True)
class Booking:
    identifier: str
    resource: str
    start: datetime
    end: datetime
    cancelled: bool = False

    def __post_init__(self):
        if not isinstance(self.identifier, str) or not self.identifier:
            raise ValueError("booking identifier must be nonempty")
        if not isinstance(self.resource, str) or not self.resource:
            raise ValueError("resource must be nonempty")
        Window(self.start, self.end)


def duration_minutes(start, end):
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        raise TypeError("boundaries must be datetimes")
    if end < start:
        raise ValueError("end must not precede start")
    return int((end - start).total_seconds() / 60)


def overlaps(left, right):
    return left.start <= right.end and right.start <= left.end


def clip_window(window, horizon):
    return Window(max(window.start, horizon.start), min(window.end, horizon.end))
