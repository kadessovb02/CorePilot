import unittest

from fixturekit.ranking import select_lowest


class Compatibility(unittest.TestCase):
    def test_lowest(self):
        low = {"score": -2}
        self.assertIs(select_lowest([{"score": 5}, low]), low)

    def test_empty(self):
        with self.assertRaises(ValueError):
            select_lowest([])

    def test_invalid(self):
        with self.assertRaises(TypeError):
            select_lowest([{"score": True}])

    def test_missing(self):
        with self.assertRaises(KeyError):
            select_lowest([{}])
