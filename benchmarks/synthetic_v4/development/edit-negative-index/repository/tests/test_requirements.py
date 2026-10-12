import unittest

from fixturekit.indexing import resolve_index


class Requirements(unittest.TestCase):
    def test_last(self):
        self.assertEqual(resolve_index(-1, 4), 3)

    def test_first_negative(self):
        self.assertEqual(resolve_index(-4, 4), 0)

    def test_one_item(self):
        self.assertEqual(resolve_index(-1, 1), 0)
