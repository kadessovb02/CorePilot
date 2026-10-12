"""CPU admission reference, excluded from model workspace."""
from pathlib import Path


def apply(root):
    root = Path(root)
    source = root / 'fixturekit/indexing.py'
    source.write_text(source.read_text().replace('position = length + index - 1 if index < 0 else index', 'position = length + index if index < 0 else index', 1))
    (root / 'tests/test_negative_index.py').write_text('import unittest\n\nfrom fixturekit.indexing import resolve_index\n\n\nclass Regression(unittest.TestCase):\n    def test_last(self):\n        self.assertEqual(resolve_index(-1, 4), 3)\n')
