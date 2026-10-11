"""Return a note with an additional label."""


def add_label(note, label):
    if not isinstance(label, str):
        raise TypeError("label must be text")
    normalized = label.strip()
    if not normalized:
        raise ValueError("label must not be empty")
    result = dict(note)
    labels = result.setdefault("labels", [])
    if not isinstance(labels, list):
        raise TypeError("labels must be a list")
    if normalized not in labels:
        labels.append(normalized)
    return result
