import unittest

from fixturekit.patterns import count_overlapping


class Requirements(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(count_overlapping("aaaa", "aa"), 3)

    def test_repeated(self):
        self.assertEqual(count_overlapping("ababa", "aba"), 2)

    def test_unicode(self):
        self.assertEqual(count_overlapping("ééé", "éé"), 2)
