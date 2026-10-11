"""Pure in-memory creation; no database, files or network."""
from .counters import next_identifier
from .parsing import parse_tags


def create_note(title, tags, notes):
    if not isinstance(title, str):
        raise TypeError("title must be text")
    title = title.strip()
    if not title:
        raise ValueError("title must not be empty")
    labels = parse_tags(tags)
    identifier = next_identifier(notes)
    note = {"id": identifier, "title": title, "labels": labels}
    return note, [*notes, note]
