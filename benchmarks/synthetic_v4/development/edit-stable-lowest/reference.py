"""CPU admission reference, excluded from model workspace."""
from pathlib import Path


def apply(root):
    root = Path(root)
    source = root / 'fixturekit/ranking.py'
    source.write_text(source.read_text().replace('if selected is None or score <= selected_score:', 'if selected is None or score < selected_score:', 1))
    (root / 'tests/test_stable_lowest.py').write_text('import unittest\n\nfrom fixturekit.ranking import select_lowest\n\n\nclass Regression(unittest.TestCase):\n    def test_tie(self):\n        first = {"score": 2, "name": "first"}\n        second = {"score": 2, "name": "second"}\n        self.assertIs(select_lowest([first, second]), first)\n')
