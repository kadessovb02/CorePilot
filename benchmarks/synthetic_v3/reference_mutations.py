"""Trusted intended changes for admission, never supplied to generation."""
from pathlib import Path

FIXES = {
    "edit-tag-normalization": ("notehub/parsing.py", [("        item = item.strip()", "        item = item.strip().casefold()"), ("tags.append(item.casefold())", "tags.append(item)")]),
    "edit-page-window": ("notehub/paging.py", [("items[offset:limit]", "items[offset:offset + limit]")]),
    "edit-label-copy": ("notehub/revision.py", [("    labels = result.setdefault(\"labels\", [])", "    labels = result.get(\"labels\", [])"), ("    if normalized not in labels:", "    labels = list(labels)\n    result[\"labels\"] = labels\n    if normalized not in labels:")]),
    "edit-text-escaping": ("notehub/export.py", [(r'return text.replace("|", "\\|").replace("\\", "\\\\").replace("\n", "\\n")', r'return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n")')]),
    "edit-sparse-identifiers": ("notehub/counters.py", [("return len(known) + 1", "return max(known, default=0) + 1")]),
    "edit-falsey-handler": ("notehub/routing.py", [("return selected or default", "return selected")]),
}

REFERENCE_TESTS = {
"edit-tag-normalization": '''import unittest
from notehub import parse_tags


class TagRegression(unittest.TestCase):
    def test_casefold_order_and_duplicates(self):
        self.assertEqual(parse_tags("B,b,Straße,STRASSE,A"), ["b", "strasse", "a"])

    def test_blanks_and_invalid(self):
        self.assertEqual(parse_tags(" , "), [])
        with self.assertRaises(TypeError):
            parse_tags(None)
''',
"edit-page-window": '''import unittest
from notehub import page


class PageRegression(unittest.TestCase):
    def test_offset_is_relative(self):
        self.assertEqual(page([0, 1, 2, 3], 2, 2), [2, 3])

    def test_tuple_and_invalid(self):
        self.assertEqual(page((1, 2), 1, 1), [2])
        with self.assertRaises(TypeError):
            page([1], True)
''',
"edit-label-copy": '''import unittest
from notehub import add_label


class LabelRegression(unittest.TestCase):
    def test_new_and_duplicate_do_not_alias(self):
        note = {"labels": ["a"]}
        for label in ("b", "a"):
            result = add_label(note, label)
            self.assertIsNot(result["labels"], note["labels"])
        self.assertEqual(note["labels"], ["a"])

    def test_missing_and_invalid(self):
        self.assertEqual(add_label({}, " x ")["labels"], ["x"])
        with self.assertRaises(ValueError):
            add_label({}, " ")
''',
"edit-text-escaping": r'''import unittest
from notehub import escape_text, render_line


class ExportRegression(unittest.TestCase):
    def test_escaping_once(self):
        self.assertEqual(escape_text("\\|\n"), "\\\\\\|\\n")

    def test_render_and_invalid(self):
        self.assertEqual(render_line({"id": 2, "title": "a|b"}), "2|a\\|b")
        with self.assertRaises(TypeError):
            escape_text(None)
''',
"edit-sparse-identifiers": '''import unittest
from notehub import next_identifier


class IdentifierRegression(unittest.TestCase):
    def test_gap_and_generator(self):
        self.assertEqual(next_identifier({"id": value} for value in (7, 2, 7)), 8)

    def test_empty_and_invalid(self):
        self.assertEqual(next_identifier([]), 1)
        with self.assertRaises(ValueError):
            next_identifier([{"id": 0}])
''',
"edit-falsey-handler": '''import unittest
from notehub import choose_handler


class FalseCallable:
    def __bool__(self):
        return False

    def __call__(self):
        return "registered"


class HandlerRegression(unittest.TestCase):
    def test_identity_and_fallback(self):
        handler = FalseCallable()
        default = lambda: "default"
        self.assertIs(choose_handler("go", {"go": handler, "default": default}), handler)
        self.assertIs(choose_handler("missing", {"default": default}), default)

    def test_invalid(self):
        with self.assertRaises(TypeError):
            choose_handler("go", {"go": None, "default": lambda: None})
''',
}


def apply_reference(case, workspace):
    filename, changes = FIXES[case["id"]]
    path = workspace / filename
    source = path.read_text()
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError("Intended mutation must match one real source span")
        source = source.replace(old, new)
    path.write_text(source)
    test = workspace / case["new_test_file"]
    test.write_text(REFERENCE_TESTS[case["id"]])


def apply_bad_fix(case, workspace):
    filename = case["target_file"]
    path = workspace / filename
    source = path.read_text()
    ident = case["id"]
    if ident == "edit-tag-normalization":
        source = source.replace("return tags", "return sorted(tags)")
    elif ident == "edit-page-window":
        source = source.replace('if isinstance(value, bool) or not isinstance(value, int):', 'if not isinstance(value, int):')
    elif ident == "edit-label-copy":
        source = source.replace("labels = list(labels)", "labels = labels")
    elif ident == "edit-text-escaping":
        source = source.replace('.replace("\\n", "\\\\n")', '')
    elif ident == "edit-sparse-identifiers":
        source = source.replace("return max(known, default=0) + 1", "return max(known, default=1)")
    elif ident == "edit-falsey-handler":
        source = source.replace("return selected", "return default")
    path.write_text(source)
