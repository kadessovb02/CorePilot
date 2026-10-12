import unittest

from fixturekit.assignments import parse_assignment


class ExternalRequirements(unittest.TestCase):
    def test_embedded(self):
        self.assertEqual(parse_assignment("mode=a=b"), ("mode", "a=b"))

    def test_adjacent(self):
        self.assertEqual(parse_assignment("a=="), ("a", "="))

    def test_spaces(self):
        self.assertEqual(parse_assignment(" key = a=b "), ("key", " a=b "))

    def test_simple(self):
        self.assertEqual(parse_assignment("x=1"), ("x", "1"))

    def test_empty_value(self):
        self.assertEqual(parse_assignment(" x ="), ("x", ""))

    def test_missing(self):
        with self.assertRaises(ValueError):
            parse_assignment("name")

    def test_invalid(self):
        with self.assertRaises(TypeError):
            parse_assignment(None)

    def test_blank_name(self):
        with self.assertRaises(ValueError):
            parse_assignment("  =value")

    def test_multi_separator(self):
        self.assertEqual(parse_assignment("a=b=c=d"), ("a", "b=c=d"))
