"""CPU admission reference, excluded from model workspace."""
from pathlib import Path


def apply(root):
    root = Path(root)
    source = root / 'fixturekit/packing.py'
    source.write_text(source.read_text().replace('return total // block_size', 'return (total + block_size - 1) // block_size', 1))
    (root / 'tests/test_ceiling_blocks.py').write_text('import unittest\n\nfrom fixturekit.packing import block_count\n\n\nclass Regression(unittest.TestCase):\n    def test_remainder(self):\n        self.assertEqual(block_count(10, 4), 3)\n')
