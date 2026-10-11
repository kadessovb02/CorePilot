from .catalog import load_product, reserve_stock
from .events import encode_event, validate_event
from .search import paginate, search_products


def summarize(products):
    rows = list(products)
    return {"product_count": len(rows),
            "total_stock": sum(product.stock for product in rows),
            "inventory_value_cents": sum(product.price_cents for product in rows),
            "low_stock_skus": sorted(product.sku for product in rows if product.stock < 2)}


class CatalogService:
    def __init__(self):
        self.products = {}
        self.events = []
        self.seen_event_ids = set()

    def add_product(self, payload):
        product = load_product(payload)
        self.products[product.sku] = product
        return product

    def reserve(self, sku, count):
        product = reserve_stock(self.products[sku], count)
        self.products[sku] = product
        return product

    def search(self, query, page=1, page_size=20):
        return paginate(search_products(list(self.products.values()), query), page, page_size)

    def summary(self):
        return summarize(self.products.values())

    def record_event(self, payload):
        event = validate_event(payload, self.seen_event_ids)
        encoded = encode_event(event)
        self.events.append(encoded)
        self.seen_event_ids.add(event["id"])
        return encoded
