import unittest

from fixturekit.prefixes import remove_prefix


class Requirements(unittest.TestCase):
    def test_later_occurrence(self):
        self.assertEqual(remove_prefix("xxabc", "abc"), "xxabc")

    def test_embedded_short(self):
        self.assertEqual(remove_prefix("abca", "a"), "bca")
        self.assertEqual(remove_prefix("xaba", "ab"), "xaba")

    def test_unicode_anchor(self):
        self.assertEqual(remove_prefix("xéé", "é"), "xéé")
