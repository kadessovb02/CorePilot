def search_products(products, query):
    lowered = query.lower()
    return [product for product in products if lowered in product.name.lower()]


def paginate(products, page=1, page_size=20):
    if not isinstance(page, int) or not isinstance(page_size, int):
        raise TypeError("page and page_size must be integers")
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be positive")
    start = page * page_size
    return list(products[start:start + page_size])
