"""Synthetic reference changes for oracle self-audits, never copied into target input."""
import ast
from pathlib import Path


MUTATIONS = {
    "edit_sku_normalization": ("inventory/catalog.py", "normalize_sku", '''def normalize_sku(value):
    if not isinstance(value, str):
        raise TypeError("sku must be a string")
    value = value.strip().upper()
    if not value:
        raise ValueError("sku must not be empty")
    return value
'''),
    "edit_exact_price": ("inventory/catalog.py", "parse_price_cents", '''def parse_price_cents(value):
    from decimal import Decimal, InvalidOperation
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise TypeError("price must be numeric")
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("price must be numeric") from None
    if not amount.is_finite() or amount < 0:
        raise ValueError("price must be finite and nonnegative")
    numerator, denominator = amount.as_integer_ratio()
    cents, remainder = divmod(numerator * 100, denominator)
    if remainder:
        raise ValueError("fractional cents are unsupported")
    return cents
'''),
    "edit_reservation_count": ("inventory/catalog.py", "reserve_stock", '''def reserve_stock(product, count):
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count <= 0 or count > product.stock:
        raise ValueError("count must be positive and available")
    return replace(product, stock=product.stock - count)
'''),
    "edit_tag_normalization": ("inventory/catalog.py", "load_product", '''def load_product(payload):
    if not isinstance(payload, dict):
        raise TypeError("product payload must be an object")
    name = payload["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a nonempty string")
    stock = payload.get("stock", 0)
    if isinstance(stock, bool) or not isinstance(stock, int) or stock < 0:
        raise ValueError("stock must be a nonnegative integer")
    incoming = payload.get("tags", ())
    if not isinstance(incoming, (list, tuple)) or any(not isinstance(tag, str) for tag in incoming):
        raise TypeError("tags must be a list or tuple of strings")
    normalized = []
    for tag in incoming:
        tag = tag.strip().casefold()
        if tag and tag not in normalized:
            normalized.append(tag)
    return Product(normalize_sku(payload["sku"]), name.strip(),
                   parse_price_cents(payload["price"]), stock, tuple(normalized))
'''),
    "edit_search_tokens": ("inventory/search.py", "search_products", '''def search_products(products, query):
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    tokens = query.casefold().split()
    return [product for product in products if all(token in " ".join(
        (product.sku, product.name, *product.tags)).casefold() for token in tokens)]
'''),
    "edit_pagination_offset": ("inventory/search.py", "paginate", '''def paginate(products, page=1, page_size=20):
    if isinstance(page, bool) or isinstance(page_size, bool) or not isinstance(page, int) or not isinstance(page_size, int):
        raise TypeError("page and page_size must be integers")
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be positive")
    start = (page - 1) * page_size
    return list(products[start:start + page_size])
'''),
    "edit_optional_token": ("inventory/client.py", "build_headers", '''def build_headers(api_token, request_id=None):
    if api_token is not None and not isinstance(api_token, str):
        raise TypeError("api_token must be a string or None")
    headers = {"Accept": "application/json"}
    token = api_token.strip() if api_token is not None else ""
    if token:
        headers["X-API-TOKEN"] = token
    if request_id is not None:
        headers["X-REQUEST-ID"] = request_id
    return headers
'''),
    "edit_exponential_backoff": ("inventory/client.py", "retry_delay", '''def retry_delay(attempt, base_delay=0.25, max_delay=4.0):
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise TypeError("attempt must be an integer")
    if attempt < 0:
        raise ValueError("attempt must be nonnegative")
    base_delay, max_delay = float(base_delay), float(max_delay)
    if not math.isfinite(base_delay) or not math.isfinite(max_delay) or min(base_delay, max_delay) < 0:
        raise ValueError("delays must be finite and nonnegative")
    delay = min(base_delay, max_delay)
    if delay == 0:
        return 0.0
    for _ in range(attempt):
        if delay >= max_delay:
            return max_delay
        delay = min(delay * 2, max_delay)
    return delay
'''),
    "edit_success_status_range": ("inventory/client.py", "decode_response", '''def decode_response(status_code, payload):
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        raise TypeError("status_code must be an integer")
    if 200 <= status_code <= 299:
        return payload
    raise RemoteError(status_code)
'''),
    "edit_event_identifier": ("inventory/events.py", "validate_event", '''def validate_event(payload, seen_ids=()):
    if not isinstance(payload, dict):
        raise TypeError("event payload must be an object")
    identifier = payload.get("id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("event id must be a nonempty string")
    identifier = identifier.strip()
    if payload.get("kind") not in {"created", "restocked"}:
        raise ValueError("unsupported event kind")
    if identifier in seen_ids:
        raise ValueError("event id was already seen")
    result = dict(payload)
    result["id"] = identifier
    return result
'''),
    "edit_canonical_event_json": ("inventory/events.py", "encode_event", '''def encode_event(event):
    return json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
'''),
    "edit_inventory_value": ("inventory/service.py", "summarize", '''def summarize(products):
    rows = list(products)
    return {"product_count": len(rows),
            "total_stock": sum(product.stock for product in rows),
            "inventory_value_cents": sum(product.price_cents * product.stock for product in rows),
            "low_stock_skus": sorted(product.sku for product in rows if product.stock < 2)}
'''),
}


def apply_reference(workspace, case_id):
    relative, function, replacement = MUTATIONS[case_id]
    path = Path(workspace) / relative
    original = path.read_text()
    node = next(node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef) and node.name == function)
    lines = original.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [replacement]
    updated = "".join(lines)
    ast.parse(updated)
    path.write_text(updated)


COUNTEREXAMPLES = {
    "edit_event_identifier": ('{"created", "restocked"}', '{"created"}'),
    "edit_exponential_backoff": (
        '    base_delay, max_delay = float(base_delay), float(max_delay)',
        '    if not isinstance(base_delay, (int, float)) or not isinstance(max_delay, (int, float)):\n'
        '        raise TypeError("numeric strings were accidentally rejected")\n'
        '    base_delay, max_delay = float(base_delay), float(max_delay)'),
    "edit_success_status_range": ('if 200 <= status_code <= 299:', 'if status_code in (200, 201, 204, 206, 299):'),
    "edit_tag_normalization": ('name.strip(),', '"Basic widget",'),
}


def apply_counterexample(workspace, case_id):
    """A plausible incomplete fix, used to prove the external oracle rejects it."""
    apply_reference(workspace, case_id)
    path = Path(workspace) / MUTATIONS[case_id][0]
    old, new = COUNTEREXAMPLES[case_id]
    source = path.read_text()
    if old not in source:
        raise ValueError("Counterexample no longer matches the reference change")
    source = source.replace(old, new)
    ast.parse(source)
    path.write_text(source)
