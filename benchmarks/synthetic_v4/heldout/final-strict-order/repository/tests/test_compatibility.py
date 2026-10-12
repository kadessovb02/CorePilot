import unittest

from fixturekit.ordering import strictly_increasing


class Compatibility(unittest.TestCase):
    def test_ordered(self):
        self.assertTrue(strictly_increasing([-3, 0, 5]))

    def test_empty(self):
        self.assertTrue(strictly_increasing([]))

    def test_descending(self):
        self.assertFalse(strictly_increasing([3, 1]))

    def test_validation(self):
        with self.assertRaises(TypeError):
            strictly_increasing([5, 1, None])
