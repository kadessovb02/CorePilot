import unittest

from fixturekit.signs import integer_sign


class ExternalRequirements(unittest.TestCase):
    def test_negative(self):
        self.assertEqual(integer_sign(-2), -1)

    def test_negative_large(self):
        self.assertEqual(integer_sign(-(10 ** 80)), -1)

    def test_negative_one(self):
        self.assertEqual(integer_sign(-1), -1)

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

    def test_positive_large(self):
        self.assertEqual(integer_sign(10 ** 80), 1)

    def test_integer_return(self):
        self.assertIs(type(integer_sign(0)), int)

    def test_float_rejected(self):
        with self.assertRaises(TypeError):
            integer_sign(1.0)
