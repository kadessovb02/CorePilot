def strictly_increasing(numbers):
    values = list(numbers)
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("values must be integers")
    return all(left <= right for left, right in zip(values, values[1:]))
