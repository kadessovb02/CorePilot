import unittest

from fixturekit.signs import integer_sign


class Requirements(unittest.TestCase):
    def test_negative(self):
        self.assertEqual(integer_sign(-2), -1)

    def test_negative_large(self):
        self.assertEqual(integer_sign(-(10 ** 80)), -1)

    def test_negative_one(self):
        self.assertEqual(integer_sign(-1), -1)
