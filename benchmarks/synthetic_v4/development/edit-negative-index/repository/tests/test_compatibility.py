import unittest

from fixturekit.indexing import resolve_index


class Compatibility(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(resolve_index(2, 4), 2)

    def test_outside(self):
        with self.assertRaises(IndexError):
            resolve_index(4, 4)

    def test_invalid(self):
        with self.assertRaises(TypeError):
            resolve_index(True, 4)

    def test_negative_length(self):
        with self.assertRaises(ValueError):
            resolve_index(0, -1)
