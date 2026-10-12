import unittest

from fixturekit.indexing import resolve_index


class ExternalRequirements(unittest.TestCase):
    def test_last(self):
        self.assertEqual(resolve_index(-1, 4), 3)

    def test_first_negative(self):
        self.assertEqual(resolve_index(-4, 4), 0)

    def test_one_item(self):
        self.assertEqual(resolve_index(-1, 1), 0)

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

    def test_outside_negative(self):
        with self.assertRaises(IndexError):
            resolve_index(-5, 4)

    def test_empty_sequence(self):
        with self.assertRaises(IndexError):
            resolve_index(0, 0)

    def test_length_bool(self):
        with self.assertRaises(TypeError):
            resolve_index(0, True)
