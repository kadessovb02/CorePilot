import unittest

from fixturekit.assignments import parse_assignment


class Compatibility(unittest.TestCase):
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
