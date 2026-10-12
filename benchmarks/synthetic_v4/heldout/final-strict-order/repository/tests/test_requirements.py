import unittest

from fixturekit.ordering import strictly_increasing


class Requirements(unittest.TestCase):
    def test_equal(self):
        self.assertFalse(strictly_increasing([1, 1]))

    def test_middle_equal(self):
        self.assertFalse(strictly_increasing([-4, 0, 0, 9]))

    def test_generator(self):
        self.assertFalse(strictly_increasing(iter([2, 2, 3])))
