import unittest

from fixturekit.ranking import select_lowest


class Requirements(unittest.TestCase):
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
