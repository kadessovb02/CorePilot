def integer_sign(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an integer")
    if value > 0:
        return 1
    return 0
