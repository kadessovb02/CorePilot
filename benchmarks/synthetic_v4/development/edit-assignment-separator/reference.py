"""CPU admission reference, excluded from model workspace."""
from pathlib import Path


def apply(root):
    root = Path(root)
    source = root / 'fixturekit/assignments.py'
    source.write_text(source.read_text().replace('    value = value.split("=")[0]\n', '', 1))
    (root / 'tests/test_assignment_separator.py').write_text('import unittest\n\nfrom fixturekit.assignments import parse_assignment\n\n\nclass Regression(unittest.TestCase):\n    def test_embedded(self):\n        self.assertEqual(parse_assignment("mode=a=b"), ("mode", "a=b"))\n')
