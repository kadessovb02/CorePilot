def parse_assignment(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if "=" not in text:
        raise ValueError("assignment needs a separator")
    name, value = text.split("=", 1)
    value = value.split("=")[0]
    name = name.strip()
    if not name:
        raise ValueError("assignment needs a name")
    return name, value
