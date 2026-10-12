import unittest

from fixturekit.packing import block_count


class ExternalRequirements(unittest.TestCase):
    def test_remainder(self):
        self.assertEqual(block_count(10, 4), 3)

    def test_below_block(self):
        self.assertEqual(block_count(1, 4), 1)

    def test_default(self):
        self.assertEqual(block_count(4097), 2)

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

    def test_large_integer(self):
        self.assertEqual(block_count(10 ** 30 + 1, 10 ** 30), 2)

    def test_negative_total(self):
        with self.assertRaises(ValueError):
            block_count(-1, 2)
