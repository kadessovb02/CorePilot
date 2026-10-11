import unittest
from decimal import Decimal
from calculator import total


class CalculatorTests(unittest.TestCase):
    def test_discount(self):
        self.assertEqual(total(["10.00", "20.00"], 10), Decimal("27.00"))

    def test_empty(self):
        self.assertEqual(total([]), Decimal("0.00"))

    def test_negative(self):
        with self.assertRaises(ValueError):
            total(["-1"])

    def test_invalid_discount(self):
        with self.assertRaises(ValueError):
            total(["10"], 101)
