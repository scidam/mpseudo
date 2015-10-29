import unittest
from mpseudo import gersgorin_bounds, pseudo
import numpy as np


class TestMpseudo(unittest.TestCase):

    def setUp(self):
        self.A_simple = np.diag(np.arange(3) + 1)
        self.A_gersh = self.A_simple.copy()
        self.A_gersh[0, 1] = 1.0
        self.A_rect = np.random.rand(4, 3)

    def test_gersgorin_bounds(self):
        self.assertEqual(gersgorin_bounds(self.A_simple), [1.0, 3.0, 0.0, 0.0])
        self.assertEqual(gersgorin_bounds(self.A_gersh), [0.0, 3.0, -1.0, 1.0])

    def test_simple(self):
        psa, X, Y = pseudo(self.A_simple, bbox=[0.0, 1.0, 0.0, 1.0], ppd=10)
        self.assertEqual(psa.shape, (10, 10))
        self.assertLessEqual(np.abs(psa[-1][-1]-
                                1.0/np.linalg.norm(self.A_simple)), 1.0e-14)

    def test_multiprocessing(self):
        psa, X, Y = pseudo(self.A_simple, ncpu=2, ppd=10)
        self.assertEqual(psa.shape, (10, 10))

# Will be uncommented when new functionality will be released.
#     def test_rectangular(self):
#         psa, X, Y = pseudo(self.A_rect, digits=20, ppd=10)
#         self.assertEqual(psa.shape, (10, 10))


if __name__ == '__main__':
    unittest.main()
