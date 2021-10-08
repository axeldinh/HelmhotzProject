from scipy.special import legendre
import numpy as np

def GaussLobattoQuadrature(num_points, a, b):
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

def Integrate(f, a, b, roots = None, weights = None):
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
        
    if type(f).__name__ == 'function':
        
        if roots is None:
            raise ValueError("f is a function handle but no values were provided for evaluation (roots = None)")
        
        xs = f(roots)
        
    else:
        xs = f
        
    integral = (b-a) * np.sum([x*w for x, w in zip(xs, weights)]) / 2
    
    return integral