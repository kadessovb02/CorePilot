from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price_cents: int
    stock: int
    tags: tuple[str, ...] = ()


def normalize_sku(value):
    if not isinstance(value, str):
        raise TypeError("sku must be a string")
    value = value.strip()
    if not value:
        raise ValueError("sku must not be empty")
    return value


def parse_price_cents(value):
    if isinstance(value, bool):
        raise TypeError("price must be numeric")
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("price must be finite and nonnegative")
    return int(amount * 100)


def load_product(payload):
    if not isinstance(payload, dict):
        raise TypeError("product payload must be an object")
    name = payload["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a nonempty string")
    stock = payload.get("stock", 0)
    if isinstance(stock, bool) or not isinstance(stock, int) or stock < 0:
        raise ValueError("stock must be a nonnegative integer")
    return Product(normalize_sku(payload["sku"]), name.strip(),
                   parse_price_cents(payload["price"]), stock, tuple(payload.get("tags", ())))


def reserve_stock(product, count):
    if not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count > product.stock:
        raise ValueError("not enough stock")
    return replace(product, stock=product.stock - count)
