import sys
sys.path.append('./src')

import quadrature_rules
import unittest
import numpy as np

class Test_Integrate1D(unittest.TestCase):

    def test_poly_legendre(self):

        num_points = 6
        f = lambda x: x**(2*num_points-3)
        a = -1
        b = 3
        roots, weights = quadrature_rules.GaussLobattoLegendreQuadrature1D(num_points, a, b)
        integral = quadrature_rules.Integrate1D(f, a, b, roots, weights)
        expected = (b**(2*num_points-2) - a**(2*num_points-2)) / (2*num_points-2)
        self.assertAlmostEqual(integral, expected, msg=f"Wrong measure in integral, should be {expected} but is {integral}")

    def test_exp_legendre(self):

        num_points = 10
        f = lambda x: np.exp(-x)
        a = -1
        b = 3
        roots, weights = quadrature_rules.GaussLobattoLegendreQuadrature1D(num_points, a, b)
        integral = quadrature_rules.Integrate1D(f, a, b, roots, weights)
        expected = np.exp(1) - np.exp(-3)
        self.assertAlmostEqual(integral, expected, msg=f"Wrong measure in integral, should be {expected} but is {integral}")

class Test_Integrate2D(unittest.TestCase):

    def test_poly_legendre(self):

        num_points = 6
        a, b = -2, 2
        c, d = 3, 4
        f = lambda x, y: x**2 * y**3
        X, Y, weights_X, weights_Y = quadrature_rules.GaussLobattoLegendreQuadrature2D(num_points, a, b, c, d)
        integral = quadrature_rules.Integrate2D(f, a, b, c, d, X, weights_X, Y, weights_Y)
        expected = 700/3
        self.assertAlmostEqual(integral, expected, msg=f"Wrong measure in integral, should be {expected} but is {integral}")

    def test_list_legendre(self):

        num_points = 6
        a, b = -2, 2
        c, d = 3, 4
        f = lambda x, y: x**2 * y**3
        X, Y, weights_X, weights_Y = quadrature_rules.GaussLobattoLegendreQuadrature2D(num_points, a, b, c, d)
        f = f(X, Y).tolist()
        integral = quadrature_rules.Integrate2D(f, a, b, c, d, X, weights_X, Y, weights_Y)
        expected = 700/3
        self.assertAlmostEqual(integral, expected, msg=f"Wrong measure in integral, should be {expected} but is {integral}")

    def test_exp_legendre(self):

        num_points = 20
        a, b = -2, 2
        c, d = 3, 4
        f = lambda x, y: np.exp(x) * y**2
        X, Y, weights_X, weights_Y = quadrature_rules.GaussLobattoLegendreQuadrature2D(num_points, a, b, c, d)
        f = f(X, Y).tolist()
        integral = quadrature_rules.Integrate2D(f, a, b, c, d, X, weights_X, Y, weights_Y)
        expected = 74 * np.sinh(2) / 3
        self.assertAlmostEqual(integral, expected, msg=f"Wrong measure in integral, should be {expected} but is {integral}")

if __name__=="__main__":
    unittest.main()