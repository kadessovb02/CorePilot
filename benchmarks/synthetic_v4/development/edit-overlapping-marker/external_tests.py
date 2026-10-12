import unittest

from fixturekit.patterns import count_overlapping


class ExternalRequirements(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(count_overlapping("aaaa", "aa"), 3)

    def test_repeated(self):
        self.assertEqual(count_overlapping("ababa", "aba"), 2)

    def test_unicode(self):
        self.assertEqual(count_overlapping("ééé", "éé"), 2)

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

    def test_case_sensitive(self):
        self.assertEqual(count_overlapping("AaA", "aa"), 0)

    def test_long_marker(self):
        self.assertEqual(count_overlapping("x", "xy"), 0)
