import unittest

from fixturekit.assignments import parse_assignment


class Requirements(unittest.TestCase):
    def test_embedded(self):
        self.assertEqual(parse_assignment("mode=a=b"), ("mode", "a=b"))

    def test_adjacent(self):
        self.assertEqual(parse_assignment("a=="), ("a", "="))

    def test_spaces(self):
        self.assertEqual(parse_assignment(" key = a=b "), ("key", " a=b "))
