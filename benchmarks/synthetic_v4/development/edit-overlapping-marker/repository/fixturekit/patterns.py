def count_overlapping(text, marker):
    if not isinstance(text, str) or not isinstance(marker, str):
        raise TypeError("text and marker must be strings")
    if not marker:
        raise ValueError("marker must not be empty")
    return text.count(marker)
