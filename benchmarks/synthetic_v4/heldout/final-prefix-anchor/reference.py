"""CPU-only admission reference; never included in model workspace."""
from pathlib import Path


def apply(root):
    root = Path(root)
    source = root / 'fixturekit/prefixes.py'
    source.write_text(source.read_text().replace('return text.replace(prefix, "", 1)', 'return text[len(prefix):] if text.startswith(prefix) else text', 1))
    (root / 'tests/test_prefix_anchor.py').write_text('import unittest\n\nfrom fixturekit.prefixes import remove_prefix\n\n\nclass Regression(unittest.TestCase):\n    def test_later_occurrence(self):\n        self.assertEqual(remove_prefix("xxabc", "abc"), "xxabc")\n')
