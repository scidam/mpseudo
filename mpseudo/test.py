import unittest
from mpseudo import gersgorin_bounds, pseudo
import numpy as np


class TestMpseudo(unittest.TestCase):

    def setUp(self):
        self.A_simple = np.diag(np.arange(3) + 1)
        self.A_gersh = self.A_simple.copy()
        self.A_gersh[0, 1] = 1.0

    def test_gersgorin_bounds(self):
        self.assertEqual(gersgorin_bounds(self.A_simple), [1.0, 3.0, 0.0, 0.0])
        self.assertEqual(gersgorin_bounds(self.A_gersh), [0.0, 3.0, -1.0, 1.0])

    def test_simple(self):
        psa, X, Y = pseudo(self.A_simple)
        self.assertEqual(psa.shape, (100, 100))

    def test_multiprocessing(self):
        psa, X, Y = pseudo(self.A_simple, ncpu=2)
        self.assertEqual(psa.shape, (100, 100))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
