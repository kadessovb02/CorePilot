import unittest

from fixturekit.patterns import count_overlapping


class Compatibility(unittest.TestCase):
    def test_ordinary(self):
        self.assertEqual(count_overlapping("abcabc", "abc"), 2)

    def test_empty(self):
        self.assertEqual(count_overlapping("", "x"), 0)

    def test_empty_marker(self):
        with self.assertRaises(ValueError):
            count_overlapping("a", "")

    def test_invalid(self):
        with self.assertRaises(TypeError):
            count_overlapping(None, "a")
