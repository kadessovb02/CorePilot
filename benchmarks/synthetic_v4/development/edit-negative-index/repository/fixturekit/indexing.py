def resolve_index(index, length):
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")
    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("length must be an integer")
    if length < 0:
        raise ValueError("length must be nonnegative")
    position = length + index - 1 if index < 0 else index
    if position < 0 or position >= length:
        raise IndexError("index is outside the sequence")
    return position
