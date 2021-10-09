from scipy.special import legendre
import numpy as np

def GaussLobattoQuadrature1D(num_points, a, b):
    '''
    Returns the num_points points and weights of the Gauss-Lobatto quadrature over [a,b].
    This method is accurate for polynomials up to degree 2num_points-3
    '''
    
    # First we recover the roots over [-1,1]
    # They corresponds to the roots of P'_n-1, P_n-1 the (n-1)th Legendre polynomial + the boundary
    poly = legendre(num_points-1)
    polyd = poly.deriv()
    roots = np.roots(polyd)
    roots = np.sort(roots)
    roots = np.concatenate([np.array([-1]), roots, 
                            np.array([1])])
    # Now we compute the weights they are given by 2/(n*(n-1)*p(x_i)^2) + 2/(n*(n-1)) on the boundary
    peval = np.polyval(poly, roots[1:-1])
    weights = (2/(peval**2)) / (num_points*(num_points-1))
    weights = np.concatenate([np.array([2/(num_points*(num_points-1))]),
                             weights,
                             np.array([2/(num_points*(num_points-1))])])
    roots = (b-a)/2 * (roots+1) + a
    return roots, weights

def GaussLobattoQuadrature2D(num_points_per_dim, a, b, c, d):
    '''
    Returns the num_points points and weights of the Gauss-Lobatto quadrature over [a,b]*[c,d].
    This method is accurate for polynomials up to degree 2num_points-3 in 1D.
    '''

    roots_X, weights_X = GaussLobattoQuadrature1D(num_points_per_dim, a, b)
    roots_Y, weights_Y = GaussLobattoQuadrature1D(num_points_per_dim, c, d)

    X, Y = np.meshgrid(roots_X, roots_Y)
    weights_X, weights_Y = np.meshgrid(weights_X, weights_X)

    return roots_X, roots_Y, weights_X, weights_Y

def Integrate1D(f, a, b, roots = None, weights = None):
    '''
    Integrates a function f over [a, b] using the given roots and weights
    Parameters:
        f (iterable or function handle): if f is an array of the same size as weights,
                                         the integral is directly computed using summation.
                                         Otherwise f is evaluated on thee roots before hand.
        a (float)                      : start of the interval
        b (float)                      : end of the interval
        roots (iterable)               : where the function needds to be evaluated (as accorded to the quadrature rule)
        weights (iterable)             : weights for the quadrature
    '''
    
    if weights is None:
        raise ValueError("No quadrature weights provided (weights = None)")

    if type(weights).__name__ != "ndarray":
        weights = np.array(weights)

    if roots is not None and type(roots).__name__ != "ndarray":
        roots = np.array(roots)
        
    if type(f).__name__ == 'function':
        if roots is None:
            raise ValueError("f is a function handle but no values were provided for evaluation (roots = None)")
        values = f(roots)
    elif type(f).__name__ != "ndarray":
        values = np.array(f)
    else:
        values = f
        
    integral = (b-a) * np.sum(values*weights) / 2
    
    return integral

def Integrate2D(f, a, b, c, d, roots_X = None,
                weights_X = None, roots_Y = None,
                weights_Y = None):
    '''
    Integrates a function f over [a, b]*[c,d] using the given roots and weights
    Parameters:
        f (iterable or function handle): if f is an array of the same size as weights,
                                         the integral is directly computed using summation.
                                         Otherwise f is evaluated on thee roots before hand.
        a (float)                      : start of the interval on x
        b (float)                      : end of the interval on x
        c (float)                      : start of the interval on y
        d (float)                      : end of the interval on y
        roots_X (iterable)             : where the function needds to be evaluated (as accorded to the quadrature rule)
        weights_X (iterable)           : weights for the quadrature
        roots_Y (iterable)             : where the function needds to be evaluated (as accorded to the quadrature rule)
        weights_Y (iterable)           : weights for the quadrature
    '''
    
    if weights_X is None or weights_Y is None:
        raise ValueError("No quadrature weights provided (weights = None)")

    # First we convert everything to numpy arrays:
    if roots_X is not None and type(roots_X).__name__ != "ndarray":
        roots_X = np.array(roots_X)
    if roots_Y is not None and type(roots_Y).__name__ != "ndarray":
        roots_Y = np.array(roots_X)
    if type(weights_X).__name__ != "ndarray":
        weights_X = np.array(roots_X)
    if type(weights_Y).__name__ != "ndarray":
        weights_Y = np.array(roots_X)
        
    if type(f).__name__ == 'function':
        if roots_X is None or roots_Y is None:
            raise ValueError("f is a function handle but no values were provided for evaluation (roots = None)")
        values = f(roots_X, roots_Y)
    elif type(f).__name__ != "ndarray":
        values = np.array(f)
    else:
        values = f

    integral = (b-a) * (d-c) * np.sum(values * weights_X * weights_Y) / 4