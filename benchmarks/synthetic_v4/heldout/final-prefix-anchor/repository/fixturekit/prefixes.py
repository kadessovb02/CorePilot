def remove_prefix(text, prefix):
    if not isinstance(text, str) or not isinstance(prefix, str):
        raise TypeError("text and prefix must be strings")
    return text.replace(prefix, "", 1)
