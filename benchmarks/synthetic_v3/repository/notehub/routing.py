"""Select a callable with a required default handler."""


def choose_handler(action, handlers):
    if not isinstance(action, str):
        raise TypeError("action must be text")
    if not action:
        raise ValueError("action must not be empty")
    default = handlers["default"]
    if not callable(default):
        raise TypeError("default must be callable")
    selected = handlers.get(action, default)
    if not callable(selected):
        raise TypeError("selected handler must be callable")
    return selected or default
