import unittest

from fixturekit.prefixes import remove_prefix


class ExternalRequirements(unittest.TestCase):
    def test_later_occurrence(self):
        self.assertEqual(remove_prefix("xxabc", "abc"), "xxabc")

    def test_embedded_short(self):
        self.assertEqual(remove_prefix("abca", "a"), "bca")
        self.assertEqual(remove_prefix("xaba", "ab"), "xaba")

    def test_unicode_anchor(self):
        self.assertEqual(remove_prefix("xéé", "é"), "xéé")

    def test_leading(self):
        self.assertEqual(remove_prefix("abcabc", "abc"), "abc")

    def test_empty_prefix(self):
        self.assertEqual(remove_prefix("abc", ""), "abc")

    def test_empty_text(self):
        self.assertEqual(remove_prefix("", "a"), "")

    def test_invalid(self):
        with self.assertRaises(TypeError):
            remove_prefix(None, "a")

    def test_case_sensitive(self):
        self.assertEqual(remove_prefix("Abc", "a"), "Abc")

    def test_same_string(self):
        self.assertEqual(remove_prefix("é", "é"), "")

    def test_prefix_type(self):
        with self.assertRaises(TypeError):
            remove_prefix("a", 1)
