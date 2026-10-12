"""CPU-only admission reference; never included in model workspace."""
from pathlib import Path


def apply(root):
    root = Path(root)
    source = root / 'fixturekit/signs.py'
    source.write_text(source.read_text().replace('    return 0\n', '    return -1 if value < 0 else 0\n', 1))
    (root / 'tests/test_integer_sign.py').write_text('import unittest\n\nfrom fixturekit.signs import integer_sign\n\n\nclass Regression(unittest.TestCase):\n    def test_negative(self):\n        self.assertEqual(integer_sign(-2), -1)\n')
