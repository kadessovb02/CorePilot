"""Choose a free numeric identifier from existing notes."""


def next_identifier(notes):
    known = []
    for note in notes:
        value = note["id"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("id must be an integer")
        if value < 1:
            raise ValueError("id must be positive")
        known.append(value)
    return len(known) + 1
