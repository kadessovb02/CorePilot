"""List pagination without changing the caller's sequence."""


def page(items, offset=0, limit=20):
    for name, value in (("offset", offset), ("limit", limit)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must not be negative")
    return list(items[offset:limit])
