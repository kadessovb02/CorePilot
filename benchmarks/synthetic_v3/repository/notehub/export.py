"""A deliberately small pipe-delimited text format."""


def escape_text(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text.replace("|", "\\|").replace("\\", "\\\\").replace("\n", "\\n")


def render_line(note):
    return f'{note["id"]}|{escape_text(note["title"])}'
