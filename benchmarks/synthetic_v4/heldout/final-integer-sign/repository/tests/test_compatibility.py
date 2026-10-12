import unittest

from fixturekit.signs import integer_sign


class Compatibility(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(integer_sign(2), 1)

    def test_zero(self):
        self.assertEqual(integer_sign(0), 0)

    def test_bool_rejected(self):
        with self.assertRaises(TypeError):
            integer_sign(True)

    def test_noninteger(self):
        with self.assertRaises(TypeError):
            integer_sign("2")
