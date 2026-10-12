import unittest

from fixturekit.ordering import strictly_increasing


class ExternalRequirements(unittest.TestCase):
    def test_equal(self):
        self.assertFalse(strictly_increasing([1, 1]))

    def test_middle_equal(self):
        self.assertFalse(strictly_increasing([-4, 0, 0, 9]))

    def test_generator(self):
        self.assertFalse(strictly_increasing(iter([2, 2, 3])))

    def test_ordered(self):
        self.assertTrue(strictly_increasing([-3, 0, 5]))

    def test_empty(self):
        self.assertTrue(strictly_increasing([]))

    def test_descending(self):
        self.assertFalse(strictly_increasing([3, 1]))

    def test_validation(self):
        with self.assertRaises(TypeError):
            strictly_increasing([5, 1, None])

    def test_singleton(self):
        self.assertTrue(strictly_increasing([2]))

    def test_bool_rejected(self):
        with self.assertRaises(TypeError):
            strictly_increasing([True])
