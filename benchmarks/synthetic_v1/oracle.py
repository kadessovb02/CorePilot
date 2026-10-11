#!/usr/bin/env python3
"""External synthetic benchmark oracle. The CLI emits exactly one JSON report."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import FrozenInstanceError
from decimal import Decimal
import hashlib
import importlib
from importlib.machinery import SourceFileLoader
import inspect
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tokenize
import unittest
from unittest.mock import patch


SUITE = Path(__file__).resolve().parent
CASE_KEYS = {"id", "prompt", "read_only", "required_reads", "oracle", "category", "max_calls", "timeout_seconds"}
TARGETS = {
    "edit_sku_normalization": "inventory/catalog.py", "edit_exact_price": "inventory/catalog.py",
    "edit_reservation_count": "inventory/catalog.py", "edit_tag_normalization": "inventory/catalog.py",
    "edit_search_tokens": "inventory/search.py", "edit_pagination_offset": "inventory/search.py",
    "edit_optional_token": "inventory/client.py", "edit_exponential_backoff": "inventory/client.py",
    "edit_success_status_range": "inventory/client.py", "edit_event_identifier": "inventory/events.py",
    "edit_canonical_event_json": "inventory/events.py", "edit_inventory_value": "inventory/service.py",
}
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".corepilot-tmp", "_benchmark_oracle"}
RUNNER_FILES = {"_benchmark_answer.json", "_benchmark_answer.txt", "_benchmark_answer.text", "_benchmark_reads.json"}
SIGNATURES = {
    "catalog": {"normalize_sku": ("value",), "parse_price_cents": ("value",),
                "load_product": ("payload",), "reserve_stock": ("product", "count")},
    "search": {"search_products": ("products", "query"), "paginate": ("products", "page", "page_size")},
    "client": {"build_headers": ("api_token", "request_id"), "retry_delay": ("attempt", "base_delay", "max_delay"),
               "decode_response": ("status_code", "payload")},
    "events": {"validate_event": ("payload", "seen_ids"), "encode_event": ("event",)},
    "service": {"summarize": ("products",)},
}


def load_cases(path=None):
    rows = json.loads(Path(path or SUITE / "cases.json").read_text())
    if not isinstance(rows, list):
        raise ValueError("cases must be a list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != CASE_KEYS:
            raise ValueError("case has missing or unsupported keys")
        for key in ("id", "prompt", "oracle", "category"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError("case string fields must be nonempty")
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,79}", row["id"]) or row["id"] in seen or row["oracle"] != row["id"]:
            raise ValueError("case id/oracle must be a unique supported identifier")
        seen.add(row["id"])
        if not isinstance(row["read_only"], bool):
            raise ValueError("read_only must be a boolean")
        if not isinstance(row["required_reads"], list) or not row["required_reads"]:
            raise ValueError("required_reads must be a nonempty list")
        for name in row["required_reads"]:
            if not isinstance(name, str) or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise ValueError("required_reads must contain safe relative paths")
            if not (SUITE / "repository" / name).is_file():
                raise ValueError("required read is outside the fixture")
        if len(set(row["required_reads"])) != len(row["required_reads"]):
            raise ValueError("required reads must not contain duplicates")
        for key, low, high in (("max_calls", 12, 24), ("timeout_seconds", 90, 150)):
            if isinstance(row[key], bool) or not isinstance(row[key], int) or not low <= row[key] <= high:
                raise ValueError("case execution budget is out of range")
        if (row["id"] in TARGETS) == row["read_only"]:
            raise ValueError("oracle and read_only mode disagree")
        if row["id"] not in TARGETS and row["id"] not in SEMANTIC_RULES:
            raise ValueError("case has an unsupported oracle")
    return rows


def source_files(root):
    """Ignore only runner infrastructure/caches; reject symlinks without following them."""
    result, violations = {}, []
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            if name in IGNORED_DIRS:
                dirs.remove(name)
            elif path.is_symlink():
                violations.append(str(path.relative_to(root)))
                dirs.remove(name)
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if directory == str(root) and (name in RUNNER_FILES or name == ".git"):
                continue
            if path.is_symlink() or path.stat().st_size > 2_000_000:
                violations.append(relative)
            else:
                result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result, violations


def style_errors(workspace, names):
    """Check the fixture's actual guidance: stdlib, snake_case and four spaces."""
    errors = []
    for name in sorted(names):
        if not name.endswith(".py") or not (workspace / name).is_file():
            continue
        try:
            data = (workspace / name).read_bytes()
            tree = ast.parse(data, filename=name)
            for token in tokenize.tokenize(io.BytesIO(data).readline):
                if token.type == tokenize.INDENT and ("\t" in token.string or len(token.string) % 4):
                    errors.append("Four-space indentation required: " + name)
                    break
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not re.fullmatch(r"[a-z_][a-z0-9_]*", node.name):
                    errors.append("snake_case function names required: " + name)
                roots = []
                if isinstance(node, ast.Import):
                    roots = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    roots = [(node.module or "").split(".")[0]]
                if any(root != "inventory" and root not in sys.stdlib_module_names for root in roots):
                    errors.append("Only standard-library and inventory imports allowed: " + name)
        except (SyntaxError, UnicodeError, tokenize.TokenError) as exc:
            errors.append("Invalid Python source: " + name + " (" + type(exc).__name__ + ")")
    return sorted(set(errors))


@contextmanager
def workspace_modules(workspace):
    old_path = list(sys.path)
    old_bytecode = sys.dont_write_bytecode
    saved = {key: value for key, value in sys.modules.items() if key == "inventory" or key.startswith("inventory.")}
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(workspace))
    sys.dont_write_bytecode = True
    original_get_code = SourceFileLoader.get_code

    def fresh_source(loader, fullname):
        if fullname == "inventory" or fullname.startswith("inventory."):
            return compile(loader.get_data(loader.path), loader.path, "exec", dont_inherit=True)
        return original_get_code(loader, fullname)

    try:
        with patch.object(SourceFileLoader, "get_code", fresh_source):
            importlib.invalidate_caches()
            modules = {"facade": importlib.import_module("inventory")}
            modules.update({name: importlib.import_module("inventory." + name) for name in SIGNATURES})
            if not Path(modules["facade"].__file__).resolve().is_relative_to(workspace):
                raise ValueError("inventory was not imported from the selected workspace")
            yield modules
    finally:
        for key in list(sys.modules):
            if key == "inventory" or key.startswith("inventory."):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode


class BehaviorChecks(unittest.TestCase):
    modules = None
    workspace = None

    def product(self, sku="SKU-A", name="Basic widget", price=300, stock=3, tags=("basic",)):
        return self.modules["catalog"].Product(sku, name, price, stock, tags)

    def test_public_api_compatibility(self):
        facade = self.modules["facade"]
        self.assertEqual(facade.__all__, ["Product", "CatalogService", "RemoteError"])
        self.assertIs(facade.Product, self.modules["catalog"].Product)
        self.assertIs(facade.CatalogService, self.modules["service"].CatalogService)
        self.assertIs(facade.RemoteError, self.modules["client"].RemoteError)
        self.assertTrue(issubclass(facade.RemoteError, RuntimeError))
        for module, functions in SIGNATURES.items():
            for name, parameters in functions.items():
                with self.subTest(function=name):
                    signature = inspect.signature(getattr(self.modules[module], name))
                    self.assertEqual(tuple(signature.parameters), parameters)
                    self.assertTrue(all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                                        for p in signature.parameters.values()))
        self.assertEqual(inspect.signature(self.modules["search"].paginate).parameters["page"].default, 1)
        self.assertEqual(inspect.signature(self.modules["search"].paginate).parameters["page_size"].default, 20)
        self.assertEqual(inspect.signature(self.modules["client"].retry_delay).parameters["base_delay"].default, 0.25)
        self.assertEqual(inspect.signature(self.modules["client"].retry_delay).parameters["max_delay"].default, 4.0)
        self.assertIsNone(inspect.signature(self.modules["client"].build_headers).parameters["request_id"].default)
        self.assertEqual(inspect.signature(self.modules["events"].validate_event).parameters["seen_ids"].default, ())
        self.assertEqual(tuple(inspect.signature(self.modules["catalog"].Product).parameters),
                         ("sku", "name", "price_cents", "stock", "tags"))
        self.assertEqual(inspect.signature(self.modules["catalog"].Product).parameters["tags"].default, ())
        value = self.product()
        with self.assertRaises(FrozenInstanceError):
            value.stock = 7

    def test_existing_correct_behaviors(self):
        catalog, search, client, events, service = (self.modules[key] for key in SIGNATURES)
        self.assertEqual(catalog.normalize_sku(" SKU-A "), "SKU-A")
        self.assertEqual(catalog.parse_price_cents("3.00"), 300)
        with self.assertRaises(ValueError):
            catalog.normalize_sku("  ")
        with self.assertRaises(TypeError):
            catalog.normalize_sku(None)
        product = self.product(stock=3)
        reserved = catalog.reserve_stock(product, 1)
        self.assertEqual((product.stock, reserved.stock), (3, 2))
        with self.assertRaises(ValueError):
            catalog.reserve_stock(product, 4)
        self.assertEqual(search.search_products([product], "WIDGET"), [product])
        self.assertEqual(search.paginate([product], 100), [])
        with self.assertRaises(ValueError):
            search.paginate([product], 0)
        self.assertEqual(client.build_headers("demo-token", "request-1"),
                         {"Accept": "application/json", "X-API-TOKEN": "demo-token", "X-REQUEST-ID": "request-1"})
        self.assertEqual(client.retry_delay(0), 0.25)
        payload = {"ok": True}
        self.assertIs(client.decode_response(200, payload), payload)
        with self.assertRaises(client.RemoteError) as error:
            client.decode_response(404, payload)
        self.assertEqual(error.exception.status_code, 404)
        self.assertEqual(str(error.exception), "remote status 404")
        event = {"id": "event-1", "kind": "created", "data": {"sku": "SKU-A"}}
        original = deepcopy(event)
        accepted = events.validate_event(event)
        self.assertEqual(accepted, original)
        self.assertIsNot(accepted, event)
        self.assertEqual(json.loads(events.encode_event(event)), event)
        self.assertEqual(event, original)
        with self.assertRaises(ValueError):
            events.validate_event(event, {"event-1"})
        with self.assertRaises(ValueError):
            events.validate_event({"id": "event-2", "kind": "unsupported"})
        catalog_service = service.CatalogService()
        loaded = catalog_service.add_product({"sku": "SKU-A", "name": "Basic widget", "price": "3.00", "stock": 1, "tags": ["basic"]})
        self.assertEqual(loaded, self.product(stock=1))
        self.assertEqual(catalog_service.summary(), {"product_count": 1, "total_stock": 1,
                         "inventory_value_cents": 300, "low_stock_skus": ["SKU-A"]})
        encoded = catalog_service.record_event(event)
        self.assertEqual(json.loads(encoded), event)
        self.assertEqual(catalog_service.events, [encoded])
        self.assertEqual(catalog_service.seen_event_ids, {"event-1"})

    def edit_sku_normalization(self):
        function = self.modules["catalog"].normalize_sku
        for value, expected in [("  sku-a \n", "SKU-A"), ("MiXeD-9", "MIXED-9"), ("straße", "STRASSE"), ("SKU-A", "SKU-A")]:
            with self.subTest(value=value):
                self.assertEqual(function(value), expected)
        for value in (None, 1, True, []):
            with self.subTest(value=value), self.assertRaises(TypeError):
                function(value)
        for value in ("", " \t\n"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                function(value)

    def edit_exact_price(self):
        function = self.modules["catalog"].parse_price_cents
        for value, expected in [("1.15", 115), ("0.29", 29), (0.29, 29), (Decimal("1.15"), 115),
                                ("1.2300", 123), (0, 0), (3, 300), ("9999999999.99", 999999999999)]:
            with self.subTest(value=str(value)):
                self.assertEqual(function(value), expected)
                self.assertIs(type(function(value)), int)
        for value in ("1.005", "-0.01", "NaN", "Infinity", float("nan"), float("inf"), "wrong"):
            with self.subTest(value=str(value)), self.assertRaises(ValueError):
                function(value)
        for value in (True, None, {}, []):
            with self.subTest(value=str(value)), self.assertRaises(TypeError):
                function(value)

    def edit_reservation_count(self):
        function = self.modules["catalog"].reserve_stock
        original = self.product(stock=3)
        for count, stock in ((1, 2), (3, 0)):
            reserved = function(original, count)
            self.assertEqual(reserved, self.product(stock=stock))
            self.assertIsNot(reserved, original)
        self.assertEqual(original.stock, 3)
        for count in (0, -1, 4):
            with self.subTest(count=count), self.assertRaises(ValueError):
                function(original, count)
        for count in (True, False, 1.0, "1", None):
            with self.subTest(count=count), self.assertRaises(TypeError):
                function(original, count)
        with self.assertRaises(ValueError):
            function(self.product(stock=0), 1)

    def edit_tag_normalization(self):
        function = self.modules["catalog"].load_product
        payload = {"sku": "SKU-A", "name": "Basic widget", "price": "3.00", "stock": 3,
                   "tags": ["  Blue ", "blue", "", " \t", "Straße", "STRASSE", "SALE"]}
        original = deepcopy(payload)
        self.assertEqual(function(payload), self.product(tags=("blue", "strasse", "sale")))
        self.assertEqual(payload, original)
        payload["tags"] = ("A", "a", "B")
        self.assertEqual(function(payload).tags, ("a", "b"))
        alternate = {"sku": " OTHER-9 ", "name": "  Travel case ", "price": "8.00", "stock": 9,
                     "tags": [" NEW ", "new"]}
        before = deepcopy(alternate)
        self.assertEqual(function(alternate), self.product("OTHER-9", "Travel case", 800, 9, ("new",)))
        self.assertEqual(alternate, before)
        del payload["tags"]
        self.assertEqual(function(payload).tags, ())
        for tags in ("blue", {"blue"}, ["blue", 1], [None]):
            with self.subTest(tags=str(tags)), self.assertRaises(TypeError):
                function({**payload, "tags": tags})

    def edit_search_tokens(self):
        function = self.modules["search"].search_products
        rows = [self.product("SKU-RED", "Red widget", tags=("sale",)),
                self.product("SKU-BLUE", "Blue widget", tags=("large",)),
                self.product("CODE-3", "Straße sign", tags=("outdoor",))]
        original = rows.copy()
        for query, expected in [("widget", rows[:2]), (" red  sale ", rows[:1]), ("sku-blue large", rows[1:2]),
                                ("STRASSE OUTDOOR", rows[2:]), ("widget unknown", []), ("\t ", rows), ("", rows)]:
            with self.subTest(query=query):
                self.assertEqual(function(iter(rows), query), expected)
        self.assertEqual(rows, original)
        self.assertIsNot(function(rows, ""), rows)
        for query in (None, 1, True):
            with self.assertRaises(TypeError):
                function(rows, query)

    def edit_pagination_offset(self):
        function = self.modules["search"].paginate
        rows = [object() for _ in range(6)]
        self.assertEqual(function(rows, 1, 2), rows[:2])
        self.assertEqual(function(tuple(rows), 2, 2), rows[2:4])
        self.assertEqual(function(rows, 3, 2), rows[4:6])
        self.assertEqual(function(rows, 4, 2), [])
        self.assertEqual(function(rows), rows)
        self.assertEqual(function([], 1, 2), [])
        self.assertIs(function(rows, 1, 1)[0], rows[0])
        for page, size in ((0, 2), (-1, 2), (1, 0), (1, -2)):
            with self.assertRaises(ValueError):
                function(rows, page, size)
        for page, size in ((True, 2), (1, False), (1.5, 2), (1, "2")):
            with self.assertRaises(TypeError):
                function(rows, page, size)

    def edit_optional_token(self):
        function = self.modules["client"].build_headers
        for token in (None, "", " \t\n"):
            with self.subTest(token=repr(token)):
                self.assertEqual(function(token), {"Accept": "application/json"})
        self.assertEqual(function(" demo token ", " request-id "), {"Accept": "application/json",
                         "X-API-TOKEN": "demo token", "X-REQUEST-ID": " request-id "})
        self.assertEqual(function(None, ""), {"Accept": "application/json", "X-REQUEST-ID": ""})
        self.assertIsNot(function(None), function(None))
        for token in (True, 1, []):
            with self.assertRaises(TypeError):
                function(token)

    def edit_exponential_backoff(self):
        function = self.modules["client"].retry_delay
        self.assertEqual([function(n) for n in range(7)], [0.25, 0.5, 1.0, 2.0, 4.0, 4.0, 4.0])
        self.assertEqual(function(3, 0.5, 3), 3.0)
        self.assertEqual(function(2, "0.5", "3"), 2.0)
        self.assertEqual(function(10_000), 4.0)
        self.assertEqual(function(10_000, 0, 4), 0.0)
        self.assertEqual(function(7, 1, 0), 0.0)
        self.assertEqual(function(0, 9, 2), 2.0)
        for value in (True, 1.5, "2", None):
            with self.assertRaises(TypeError):
                function(value)
        for args in ((-1,), (1, -1, 4), (1, 1, -4), (1, float("inf"), 4), (1, 1, float("nan"))):
            with self.assertRaises(ValueError):
                function(*args)

    def edit_success_status_range(self):
        function, error = self.modules["client"].decode_response, self.modules["client"].RemoteError
        payload = {"items": []}
        for status in range(200, 300):
            with self.subTest(status=status):
                self.assertIs(function(status, payload), payload)
        self.assertIsNone(function(204, None))
        for status in (-1, 100, 199, 300, 404, 500):
            with self.subTest(status=status), self.assertRaises(error) as context:
                function(status, payload)
            self.assertEqual(context.exception.status_code, status)
            self.assertEqual(str(context.exception), f"remote status {status}")
        for status in (True, "200", 200.0, None):
            with self.assertRaises(TypeError):
                function(status, payload)

    def edit_event_identifier(self):
        function = self.modules["events"].validate_event
        payload = {"id": " event-1 \t", "kind": "created", "extra": {"count": 1}}
        original = deepcopy(payload)
        result = function(payload, {"event-0"})
        self.assertEqual(result["id"], "event-1")
        self.assertIs(result["extra"], payload["extra"])
        self.assertIsNot(result, payload)
        self.assertEqual(payload, original)
        restocked = {"id": " restock-2 ", "kind": "restocked", "extra": {"count": 2}}
        restocked_before = deepcopy(restocked)
        restocked_result = function(restocked, {"event-1"})
        self.assertEqual(restocked_result, {**restocked, "id": "restock-2"})
        self.assertIs(restocked_result["extra"], restocked["extra"])
        self.assertIsNot(restocked_result, restocked)
        self.assertEqual(restocked, restocked_before)
        for event in ({"id": " \t", "kind": "created"}, payload, {"id": 9, "kind": "created"}):
            with self.subTest(event=repr(event)), self.assertRaises(ValueError):
                function(event, {"event-1"})
        instance = self.modules["service"].CatalogService()
        instance.record_event(payload)
        self.assertEqual(json.loads(instance.record_event(restocked)), restocked_result)
        with self.assertRaises(ValueError):
            instance.record_event({"id": "event-1", "kind": "restocked"})
        with self.assertRaises(ValueError):
            instance.record_event({"id": " restock-2 ", "kind": "restocked"})
        self.assertEqual(instance.seen_event_ids, {"event-1", "restock-2"})
        self.assertEqual(len(instance.events), 2)

    def edit_canonical_event_json(self):
        function = self.modules["events"].encode_event
        value = {"z": "café", "a": {"y": [1, None, True], "b": "★"}}
        original = deepcopy(value)
        expected = '{"a":{"b":"★","y":[1,null,true]},"z":"café"}'
        self.assertEqual(function(value), expected)
        self.assertEqual(function({"a": value["a"], "z": value["z"]}), expected)
        self.assertEqual(json.loads(function(value)), original)
        self.assertEqual(value, original)
        self.assertEqual(function({}), "{}")
        with self.assertRaises(TypeError):
            function({"not_json": {1, 2}})

    def edit_inventory_value(self):
        function = self.modules["service"].summarize
        rows = [self.product("SKU-C", price=123, stock=2), self.product("SKU-A", price=500, stock=3),
                self.product("SKU-Z", price=999, stock=0), self.product("SKU-B", price=7, stock=1)]
        original = rows.copy()
        expected = {"product_count": 4, "total_stock": 6, "inventory_value_cents": 1753,
                    "low_stock_skus": ["SKU-B", "SKU-Z"]}
        self.assertEqual(function(iter(rows)), expected)
        self.assertEqual(function(rows), expected)
        self.assertEqual(rows, original)
        self.assertIs(type(function(rows)["inventory_value_cents"]), int)
        self.assertEqual(function(iter(())), {"product_count": 0, "total_stock": 0,
                         "inventory_value_cents": 0, "low_stock_skus": []})

    def inspection_source_facts(self):
        """Verify reference facts independently of an answer's vocabulary."""
        case = self.case_id
        catalog, search, client, events, service = (self.modules[key] for key in SIGNATURES)
        if case == "inspect_loader_flow":
            self.assertIn("float(value)", inspect.getsource(catalog.parse_price_cents))
            instance = service.CatalogService()
            value = instance.add_product({"sku": "SKU-A", "name": "Basic", "price": "3.00"})
            self.assertIsInstance(value, catalog.Product)
            self.assertIs(instance.products["SKU-A"], value)
        elif case == "inspect_record_immutability":
            payload = {"sku": "SKU-A", "name": "Basic", "price": "3.00", "tags": ["basic"]}
            before = deepcopy(payload)
            value = catalog.load_product(payload)
            self.assertEqual(before, payload)
            self.assertIs(type(value.tags), tuple)
            self.assertTrue(catalog.Product.__dataclass_params__.frozen)
        elif case == "inspect_search_pagination":
            product = self.product("CODE-1", "Green widget", tags=("forest",))
            self.assertEqual(search.search_products([product], "CODE-1"), [])
            self.assertEqual(search.search_products([product], "forest"), [])
            self.assertEqual(search.search_products([product], "widget green"), [])
            self.assertEqual(search.paginate(list(range(6)), 1, 2), [2, 3])
        elif case == "inspect_transport_boundary":
            source = ast.parse(inspect.getsource(client))
            calls = [ast.unparse(node.func) for node in ast.walk(source) if isinstance(node, ast.Call)]
            self.assertFalse(any(name.endswith((".sleep", ".get", ".post", ".request")) for name in calls))
            self.assertEqual(client.retry_delay(2), 0.75)
            with self.assertRaises(client.RemoteError):
                client.decode_response(201, {})
        elif case == "inspect_event_flow":
            instance = service.CatalogService()
            with self.assertRaises(ValueError):
                instance.record_event({"id": "bad", "kind": "unsupported"})
            self.assertEqual((instance.events, instance.seen_event_ids), ([], set()))
            instance.record_event({"id": "one", "kind": "created"})
            instance.record_event({"id": "two", "kind": "restocked"})
            with self.assertRaises(ValueError):
                instance.record_event({"id": "one", "kind": "created"})
            self.assertEqual((len(instance.events), instance.seen_event_ids), (2, {"one", "two"}))
        elif case == "inspect_persistence_claim":
            instance = service.CatalogService()
            self.assertEqual((instance.products, instance.events, instance.seen_event_ids), ({}, [], set()))
            imports = []
            for name in ("catalog", "search", "client", "events", "service"):
                for node in ast.walk(ast.parse(inspect.getsource(self.modules[name]))):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        imports.append(node.module or "")
            self.assertFalse(any(name.startswith(("django", "psycopg", "sqlite3", "sqlalchemy")) for name in imports))
        elif case == "inspect_token_edge_cases":
            self.assertNotIn("X-API-TOKEN", client.build_headers(None))
            self.assertEqual(client.build_headers("")["X-API-TOKEN"], "")
            self.assertEqual(client.build_headers(" \t")["X-API-TOKEN"], " \t")
            self.assertEqual(client.build_headers(" demo ", "request")["X-API-TOKEN"], " demo ")
        elif case == "inspect_public_facade":
            self.assertEqual(self.modules["facade"].__all__, ["Product", "CatalogService", "RemoteError"])
            self.assertTrue(issubclass(client.RemoteError, RuntimeError))
            self.assertEqual(client.RemoteError(503).status_code, 503)
        else:
            self.fail("Unsupported inspection oracle")


SEMANTIC_RULES = {
    "inspect_loader_flow": [("loader call flow", r"add_product.{0,180}load_product|load_product.{0,180}add_product"),
        ("frozen record", r"frozen|заморож|неизменяем"), ("dictionary state", r"(?:dict|dictionary|словар).{0,100}(?:sku|products)|products.{0,100}(?:dict|dictionary|словар)"),
        ("float conversion limitation", r"float.{0,160}(?:int|truncat|precision|decimal|точност)")],
    "inspect_record_immutability": [("payload remains unchanged", r"(?:not|doesn't|without|не).{0,80}(?:mutat|измен)|payload.{0,100}(?:unchanged|не измен)"),
        ("frozen record", r"frozen|заморож|неизменяем"), ("tuple tags", r"tags.{0,70}tuple|tuple.{0,70}tags|тег.{0,70}кортеж"),
        ("shallow freezing boundary", r"shallow|recursiv|глубок|рекурсив|поверхност")],
    "inspect_search_pagination": [("filter then page", r"search_products.{0,160}paginate"),
        ("name-only substring", r"(?:substring|подстрок).{0,100}name|name.{0,100}(?:substring|подстрок)"),
        ("current wrong offset", r"page\s*\*\s*page_size|(?:offset|start|смещ).{0,60}(?:page|2)"),
        ("first two skipped", r"(?:skip|пропуск).{0,60}(?:first two|first 2|первые два|первых двух)|(?:index|indexes|indices|индекс).{0,40}2.{0,20}3")],
    "inspect_transport_boundary": [("no transport", r"(?:no|not|without|не).{0,100}(?:http|network|transport|сетев)"),
        ("no sleeping", r"(?:no|not|without|не).{0,120}(?:sleep|sleeping|спит|ожида)"),
        ("linear delay", r"linear|линей|attempt\s*\+\s*1"), ("only 200 succeeds", r"(?:only|только).{0,70}200|200.{0,70}(?:only|только)")],
    "inspect_event_flow": [("validation before recording", r"validat.{0,120}(?:before|then|prior)|before.{0,80}(?:encod|append)|валидац.{0,100}(?:до|перед)"),
        ("created kind", r"\bcreated\b"), ("restocked kind", r"\brestocked\b"),
        ("set duplicate tracking", r"seen_event_ids.{0,70}set|set.{0,70}seen_event_ids|множеств.{0,90}(?:id|идентифик)"),
        ("rejection keeps state", r"(?:reject|invalid|отклон).{0,140}(?:neither|not|unchanged|не)"),
        ("in-memory storage", r"in.memory|per.instance|в памяти")],
    "inspect_persistence_claim": [("no claimed database layer", r"(?:no|not|without|нет|не).{0,100}(?:django|postgres|database|persist|durab)|(?:django|postgres).{0,80}(?:absent|not|нет|не)"),
        ("products dictionary", r"products.{0,120}(?:dict|dictionary|словар)"), ("events list", r"events.{0,90}(?:list|список)"),
        ("seen ids set", r"seen_event_ids.{0,80}(?:set|множеств)"),
        ("scope of inspected evidence", r"external.{0,150}(?:not|unsupported|unverified|не)|(?:only|current).{0,80}fixture|внешн.{0,120}(?:не|нет)")],
    "inspect_token_edge_cases": [("None omission", r"none.{0,80}(?:omit|exclude|no header|не|отсут)"),
        ("empty still included", r"empty.{0,130}(?:include|header|still)|пуст.{0,90}(?:перед|добав|содерж)"),
        ("whitespace included", r"whitespace.{0,130}(?:include|header|original|still)|пробел.{0,100}(?:перед|добав|содерж)"),
        ("no trim", r"(?:not|doesn't|without|не).{0,40}(?:strip|trim|обрез)|(?:strip|trim).{0,40}(?:not|не)"),
        ("Accept preserved", r"accept.{0,80}application/json"), ("request-id copied", r"request.?id.{0,90}(?:cop|header|x-request-id|перед)")],
    "inspect_public_facade": [("declared exports", r"__all__|public.{0,50}export|публичн.{0,50}экспорт"),
        ("Product export", r"\bProduct\b"), ("CatalogService export", r"\bCatalogService\b"),
        ("RemoteError export", r"\bRemoteError\b"), ("RuntimeError ancestry", r"(?:inherits|subclass|base|наслед).{0,70}RuntimeError|RuntimeError.{0,70}(?:base|наслед)"),
        ("status retained", r"(?:retain|store|attribute|сохран|поле).{0,70}status_code|status_code.{0,70}(?:retain|store|attribute|сохран|поле)"),
        ("no third-party dependencies", r"standard.library|no.{0,40}third.party|стандартн.{0,40}библиот")],
}
FORBIDDEN = {
    "inspect_loader_flow": [r"(?:uses|implements)\s+(?:exact\s+)?(?:Decimal|decimal arithmetic)"],
    "inspect_record_immutability": [r"(?:mutates|modifies)\s+(?:the\s+)?(?:caller|payload)"],
    "inspect_search_pagination": [r"searches\s+(?:the\s+)?(?:SKU|tags)"],
    "inspect_transport_boundary": [r"sends?\s+(?:an?\s+)?HTTP\s+request", r"sleeps?\s+(?:between|before)\s+retries"],
    "inspect_persistence_claim": [r"uses\s+(?:Django|PostgreSQL)"],
    "inspect_token_edge_cases": [r"tokens?\s+(?:are|is)\s+(?:already\s+)?(?:trimmed|stripped)"],
}


def semantic_grade(case_id, answer):
    missing = [label for label, pattern in SEMANTIC_RULES[case_id]
               if not re.search(pattern, answer, re.I | re.S)]
    contradictions = []
    for pattern in FORBIDDEN.get(case_id, []):
        for match in re.finditer(pattern, answer, re.I):
            prefix = re.split(r"[.!?\n]", answer[:match.start()])[-1][-120:]
            directly_negated = re.search(r"\b(?:does\s+not|doesn't|do\s+not|don't|never|not|neither|не)\s+(?:actually\s+)?$", prefix, re.I)
            coordinated_negation = re.search(
                r"\b(?:(?:never|does\s+not|doesn't)\s+send(?:s)?(?:\s+(?:an?|any))?\s+HTTP\s+requests?\s+or|"
                r"neither\s+send(?:s)?(?:\s+(?:an?|any))?\s+HTTP\s+requests?\s+nor)\s+$", prefix, re.I)
            # A quoted assertion is exempt only when its same sentence explicitly refutes it.
            sentence_tail = re.split(r"[.!?\n]", answer[match.end():], maxsplit=1)[0]
            refuted_quote = (re.search(r"\b(?:claim|assertion)\b.{0,40}$", prefix, re.I)
                             and re.search(r"\b(?:is|was|are)\s+(?:clearly\s+)?(?:false|incorrect|unsupported|untrue)\b", sentence_tail, re.I))
            if not directly_negated and not coordinated_negation and not refuted_quote:
                contradictions.append("Unsupported affirmative statement")
    return {"passed": not missing and not contradictions, "missing_criteria": missing,
            "contradictions": contradictions,
            "limitation": "Bounded lexical/semantic proxy; inspect the answer and successful read trace manually."}


def evaluate(case_id, workspace, answer=None, observed_reads=(), *, facts_only=False):
    cases = {case["id"]: case for case in load_cases()}
    if case_id not in cases:
        raise ValueError("Unknown case id")
    case = cases[case_id]
    workspace = Path(workspace).resolve()
    baseline, baseline_errors = source_files(SUITE / "repository")
    actual, integrity_errors = source_files(workspace)
    initial_source_errors = list(integrity_errors)
    if baseline_errors:
        raise ValueError("The benchmark baseline is invalid")
    allowed = set() if case["read_only"] else {TARGETS[case_id]}
    changed = [name for name, digest in baseline.items() if actual.get(name) != digest]
    unexpected = sorted(set(actual) - set(baseline))
    allowed_tests = lambda name: not case["read_only"] and re.fullmatch(r"tests/(?:__init__|test_[a-zA-Z0-9_]+)\.py", name)
    integrity_errors += ["Unexpected change: " + name for name in changed if name not in allowed]
    integrity_errors += ["Unexpected file: " + name for name in unexpected if not allowed_tests(name)]
    guidance_errors = style_errors(workspace, set(changed) | set(unexpected))
    reads = set()
    for path in observed_reads:
        if isinstance(path, str) and not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts:
            reads.add(PurePosixPath(path).as_posix())
    missing_reads = sorted(set(case["required_reads"]) - reads)
    report = {"schema_version": 1, "case_id": case_id, "oracle": case["oracle"],
              "oracle_kind": "inspection_source_facts" if facts_only else "inspection_semantic_proxy" if case["read_only"] else "unittest",
              "passed": False, "integrity_errors": integrity_errors, "changed_files": sorted(changed),
              "style_errors": guidance_errors,
              "required_reads_missing": missing_reads, "tests_run": 0, "failure_count": 0, "error_count": 0,
              "failures": [], "errors": []}
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        try:
            with workspace_modules(workspace) as modules:
                scoped = type("ScopedBehaviorChecks", (BehaviorChecks,), {"modules": modules, "workspace": workspace, "case_id": case_id})
                names = ["test_public_api_compatibility", "test_existing_correct_behaviors",
                         "inspection_source_facts" if case["read_only"] else case_id]
                result = unittest.TestResult()
                unittest.TestSuite(scoped(name) for name in names).run(result)
                report["tests_run"] = result.testsRun
                for key in ("failures", "errors"):
                    items = getattr(result, key)
                    report["failure_count" if key == "failures" else "error_count"] = len(items)
                    report[key] = [{"test": getattr(item[0], "test_case", item[0])._testMethodName, "detail": item[1].replace(str(workspace), "<workspace>").replace(str(SUITE), "<oracle>")[-600:]}
                                   for item in items[:3]]
                report["passed"] = result.wasSuccessful() and not integrity_errors and not guidance_errors
        except Exception as exc:
            detail = str(exc).replace(str(workspace), "<workspace>").replace(str(SUITE), "<oracle>")[:500]
            report["errors"].append({"test": "import_workspace", "detail": type(exc).__name__ + ": " + detail})
            report["error_count"] += 1
    after, after_errors = source_files(workspace)
    if after != actual or after_errors != initial_source_errors:
        report["integrity_errors"].append("Workspace source changed while the oracle executed")
        report["passed"] = False
    if case["read_only"] and not facts_only:
        report["semantic"] = semantic_grade(case_id, answer or "")
        report["passed"] = report["passed"] and not missing_reads and report["semantic"]["passed"]
    if facts_only and not case["read_only"]:
        raise ValueError("facts_only is reserved for inspection baseline audits")
    report["captured_output_chars"] = len(output.getvalue())
    report["read_gate_enforced"] = case["read_only"] and not facts_only
    report["fixture_fingerprint"] = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
    return report


def bounded_text(path, maximum=20_000):
    path = Path(path)
    if path.stat().st_size > maximum * 4:
        raise ValueError("Answer input is too large")
    value = path.read_text(encoding="utf-8")
    if len(value) > maximum:
        raise ValueError("Answer input is too large")
    # A trusted runner may save raw text or {"answer": "..."}.
    if path.suffix == ".json":
        parsed = json.loads(value)
        value = parsed.get("answer") if isinstance(parsed, dict) else parsed
        if not isinstance(value, str):
            raise ValueError("Answer JSON must be a string or an object containing answer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--answer-file")
    parser.add_argument("--reads-file", help="JSON array of paths from actual successful read_file results")
    parser.add_argument("--facts-only", action="store_true", help="Audit inspection source facts; this is not task success")
    args = parser.parse_args()
    try:
        answer = bounded_text(args.answer_file) if args.answer_file else None
        reads = json.loads(Path(args.reads_file).read_text()) if args.reads_file else []
        if not isinstance(reads, list) or len(reads) > 200:
            raise ValueError("reads-file must contain at most 200 paths")
        report = evaluate(args.case, args.workspace, answer, reads, facts_only=args.facts_only)
    except Exception as exc:
        report = {"schema_version": 1, "case_id": args.case, "passed": False,
                  "oracle_kind": "configuration_error", "errors": [{"detail": type(exc).__name__ + ": " + str(exc)[:500]}]}
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
