import unittest
from .mpseudo import gersgorin_bounds, pseudo, eigen_bounds
import numpy as np


class TestMpseudo(unittest.TestCase):

    def setUp(self):
        self.A_simple = np.diag(np.arange(3) + 1)
        self.A_gersh = self.A_simple.copy()
        self.A_gersh[0, 1] = 1.0
        self.A_rect = np.random.rand(4, 3)
        self.A_rect_inv = np.random.rand(3, 4)
        self.A_rect_complex = np.random.rand(3, 4) +\
            (1j) * np.random.rand(3, 4)

    def default_asserts(self, psa, X, Y):
        self.assertEqual(psa.shape, (10, 10))
        self.assertEqual(X.shape, (10, 10))
        self.assertEqual(Y.shape, (10, 10))

    def test_gersgorin_bounds(self):
        self.assertEqual(gersgorin_bounds(self.A_simple), [1.0, 3.0, 0.0, 0.0])
        self.assertEqual(gersgorin_bounds(self.A_gersh), [0.0, 3.0, -1.0, 1.0])

    def test_eigen_bounds(self):
        ebn10 = eigen_bounds(self.A_simple)
        self.assertEqual(ebn10, [0.8, 3.2, -0.2, 0.2])
        ebn20 = eigen_bounds(self.A_simple, percent=0.2)
        self.assertEqual(ebn20, [0.6, 3.4, -0.4, 0.4])

    def test_invoke_eigen_bounds(self):
        psa, X, Y = pseudo(self.A_rect_inv, digits=10,
                           ppd=10, bbox=eigen_bounds
                           )
        self.default_asserts(psa, X, Y)

    def test_simple(self):
        psa, X, Y = pseudo(self.A_simple, bbox=[0.0, 1.0, 0.0, 1.0], ppd=10)
        self.default_asserts(psa, X, Y)
        self.assertLessEqual(np.abs(psa[-1][-1] -
                             1.0 / np.linalg.norm(self.A_simple)), 1.0e-14)

    def test_multiprocessing(self):
        psa, X, Y = pseudo(self.A_simple, ncpu=2, ppd=10)
        self.default_asserts(psa, X, Y)

    def test_rectangular(self):
        psa, X, Y = pseudo(self.A_rect, digits=20, ppd=10)
        self.default_asserts(psa, X, Y)

    def test_rectangular_inv(self):
        psa, X, Y = pseudo(self.A_rect_inv, digits=10, ppd=10)
        self.default_asserts(psa, X, Y)

    def test_rectangular_gershgorin_bounds_str(self):
        psa, X, Y = pseudo(self.A_rect, digits=10, ppd=10, bbox='Nothing')
        self.default_asserts(psa, X, Y)

    def test_rectangular_gershgorin_bounds_none(self):
        psa, X, Y = pseudo(self.A_rect, digits=10, ppd=10, bbox=None)
        self.default_asserts(psa, X, Y)

    def test_simple_integer_bbox(self):
        psa, X, Y = pseudo(self.A_simple, digits=10, ppd=10, bbox=[1, 2, 3, 4])
        self.default_asserts(psa, X, Y)

    def test_rectangular_gershgorin_callable(self):
        def _bbox_func(*args):
            return [0, 1, 0, 1]
        psa, X, Y = pseudo(self.A_rect,
                           digits=10, ppd=10, bbox=_bbox_func, ncpu=2)
        self.assertEqual(psa.shape, (10, 10))
        self.assertEqual(X.ravel()[-1], 1.0)
        self.assertEqual(Y.ravel()[-1], 1.0)

    def test_complex_matrix(self):
        psa, X, Y = pseudo(self.A_rect_complex, digits=10, ppd=10, bbox=None)
        self.default_asserts(psa, X, Y)

if __name__ == '__main__':
    unittest.main()
