"""CPU admission reference, excluded from model workspace."""
from pathlib import Path


def apply(root):
    root = Path(root)
    source = root / 'fixturekit/ordering.py'
    source.write_text(source.read_text().replace('return all(left <= right for left, right in zip(values, values[1:]))', 'return all(left < right for left, right in zip(values, values[1:]))', 1))
    (root / 'tests/test_strict_order.py').write_text('import unittest\n\nfrom fixturekit.ordering import strictly_increasing\n\n\nclass Regression(unittest.TestCase):\n    def test_equal(self):\n        self.assertFalse(strictly_increasing([1, 1]))\n')
