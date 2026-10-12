import unittest

from fixturekit.packing import block_count


class Compatibility(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(block_count(12, 4), 3)

    def test_zero(self):
        self.assertEqual(block_count(0), 0)

    def test_invalid(self):
        with self.assertRaises(TypeError):
            block_count(True, 2)

    def test_bad_size(self):
        with self.assertRaises(ValueError):
            block_count(2, 0)
