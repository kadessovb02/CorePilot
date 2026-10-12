import unittest

from fixturekit.prefixes import remove_prefix


class Compatibility(unittest.TestCase):
    def test_leading(self):
        self.assertEqual(remove_prefix("abcabc", "abc"), "abc")

    def test_empty_prefix(self):
        self.assertEqual(remove_prefix("abc", ""), "abc")

    def test_empty_text(self):
        self.assertEqual(remove_prefix("", "a"), "")

    def test_invalid(self):
        with self.assertRaises(TypeError):
            remove_prefix(None, "a")
