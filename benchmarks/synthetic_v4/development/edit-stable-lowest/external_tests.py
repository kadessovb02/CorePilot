import unittest

from fixturekit.ranking import select_lowest


class ExternalRequirements(unittest.TestCase):
    def test_tie(self):
        first = {"score": 2, "name": "first"}
        second = {"score": 2, "name": "second"}
        self.assertIs(select_lowest([first, second]), first)

    def test_after_lower(self):
        first = {"score": -3}
        second = {"score": -3}
        self.assertIs(select_lowest([{"score": 8}, first, second]), first)

    def test_generator_tie(self):
        first = {"score": 4}
        self.assertIs(select_lowest(iter([first, {"score": 4}])), first)

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

    def test_late_invalid(self):
        with self.assertRaises(TypeError):
            select_lowest([{"score": 1}, {"score": "2"}])

    def test_immutability(self):
        item = {"score": 2, "other": [1]}
        original = dict(item)
        self.assertIs(select_lowest([item]), item)
        self.assertEqual(item, original)
