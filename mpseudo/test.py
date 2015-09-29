import unittest
from mpseudo import gersgorin_bounds, pseudo
import numpy as np

class TestStringMethods(unittest.TestCase):
    
    def setUp(self):
        self.A_simple = np.diag(np.arange(3)+1)

    def test_simple(self):
        psa, X, Y = pseudo(self.A_simple)
        self.assertEqual(psa.shape, (100,100))

    def test_multiprocessing(self):
        psa, X, Y = pseudo(self.A_simple, ncpu=2)
        self.assertEqual(psa.shape, (100,100))

    def tearDown(self):
        pass
    
    
    
if __name__ == '__main__':
    unittest.main()