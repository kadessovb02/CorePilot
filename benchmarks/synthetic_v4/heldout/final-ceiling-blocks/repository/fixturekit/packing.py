def block_count(total, block_size=4096):
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer")
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("block_size must be an integer")
    if total < 0 or block_size <= 0:
        raise ValueError("invalid sizes")
    return total // block_size
