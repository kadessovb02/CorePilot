"""Comma-separated tag parsing."""


def parse_tags(value):
    if not isinstance(value, str):
        raise TypeError("tags must be text")
    tags = []
    seen = set()
    for item in value.split(","):
        item = item.strip()
        if item and item not in seen:
            tags.append(item.casefold())
            seen.add(item)
    return tags
