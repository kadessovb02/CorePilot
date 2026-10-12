import unittest

from fixturekit.packing import block_count


class Requirements(unittest.TestCase):
    def test_remainder(self):
        self.assertEqual(block_count(10, 4), 3)

    def test_below_block(self):
        self.assertEqual(block_count(1, 4), 1)

    def test_default(self):
        self.assertEqual(block_count(4097), 2)
