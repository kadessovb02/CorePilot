"""CPU admission reference, excluded from model workspace."""
from pathlib import Path


def apply(root):
    root = Path(root)
    source = root / 'fixturekit/patterns.py'
    source.write_text(source.read_text().replace('return text.count(marker)', 'return sum(text.startswith(marker, index) for index in range(len(text)))', 1))
    (root / 'tests/test_overlapping_marker.py').write_text('import unittest\n\nfrom fixturekit.patterns import count_overlapping\n\n\nclass Regression(unittest.TestCase):\n    def test_overlap(self):\n        self.assertEqual(count_overlapping("aaaa", "aa"), 3)\n')
