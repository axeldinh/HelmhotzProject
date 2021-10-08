import sys
sys.path.append('./src')

import quadrature_rules
import unittest
import numpy as np

class Test_Integrate(unittest.TestCase):

    def test_poly(self):

        num_points = 6
        f = lambda x: x**(2*num_points-3)
        a = -1
        b = 3
        roots, weights = quadrature_rules.GaussLobattoQuadrature(num_points, a, b)
        integral = quadrature_rules.Integrate(f, a, b, roots, weights)
        expected = (b**(2*num_points-2) - a**(2*num_points-2)) / (2*num_points-2)
        self.assertAlmostEqual(integral, expected, msg=f"Wrong measure in integral, should be {expected} but is {integral}")

    def test_exp(self):

        num_points = 10
        f = lambda x: np.exp(-x)
        a = -1
        b = 3
        roots, weights = quadrature_rules.GaussLobattoQuadrature(num_points, a, b)
        integral = quadrature_rules.Integrate(f, a, b, roots, weights)
        expected = np.exp(1) - np.exp(-3)
        self.assertAlmostEqual(integral, expected, msg=f"Wrong measure in integral, should be {expected} but is {integral}")

if __name__=="__main__":
    unittest.main()