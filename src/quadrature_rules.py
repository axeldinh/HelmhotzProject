from scipy.special import legendre, roots_jacobi, jacobi, gamma
import numpy as np
import torch


# GAUSS-LOBATTO-LEGENDRE QUADRATURE
##################################################

def GaussLobattoLegendreQuadrature1D(num_points, a, b):
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
    X = (b-a)/2 * (roots+1) + a
    return X, weights

def GaussLobattoLegendreQuadrature2D(num_points_per_dim, a, b, c, d):
    '''
    Returns the num_points points and weights of the Gauss-Lobatto quadrature over [a,b]*[c,d].
    This method is accurate for polynomials up to degree 2num_points-3 in 1D.
    '''

    roots_X, weights_X = GaussLobattoLegendreQuadrature1D(num_points_per_dim, a, b)
    roots_Y, weights_Y = GaussLobattoLegendreQuadrature1D(num_points_per_dim, c, d)

    X, Y = np.meshgrid(roots_X, roots_Y)
    weights_X, weights_Y = np.meshgrid(weights_X, weights_Y)

    return X, Y, weights_X, weights_Y

# GAUSS-LOBATTO-JACOBI QUADRATURE
##################################################

def Jacobi(n,alpha,beta,x):
    '''
    Recursive generation of the Jacobi polynomial of order n
    '''
    x=np.array(x)
    return (jacobi(n,alpha,beta)(x))
    
def GaussLobattoJacobiWeights(Q: int, alpha = 0,beta = 0):
    '''
    Weight coefficients
    '''
    W = []
    X = roots_jacobi(Q-2,alpha+1,beta+1)[0]
    if alpha == 0 and beta==0:
        W = 2/( (Q-1)*(Q)*(Jacobi(Q-1,0,0,X)**2) )
        Wl = 2/( (Q-1)*(Q)*(Jacobi(Q-1,0,0,-1)**2) )
        Wr = 2/( (Q-1)*(Q)*(Jacobi(Q-1,0,0,1)**2) )
    else:
        W = 2**(alpha+beta+1)*gamma(alpha+Q)*gamma(beta+Q)/( (Q-1)*gamma(Q)*gamma(alpha+beta+Q+1)*(Jacobi(Q-1,alpha,beta,X)**2) )
        Wl = (beta+1)*2**(alpha+beta+1)*gamma(alpha+Q)*gamma(beta+Q)/( (Q-1)*gamma(Q)*gamma(alpha+beta+Q+1)*(Jacobi(Q-1,alpha,beta,-1)**2) )
        Wr = (alpha+1)*2**(alpha+beta+1)*gamma(alpha+Q)*gamma(beta+Q)/( (Q-1)*gamma(Q)*gamma(alpha+beta+Q+1)*(Jacobi(Q-1,alpha,beta,1)**2) )
    W = np.append(W , Wr)
    W = np.append(Wl , W)
    X = np.append(X , 1)
    X = np.append(-1 , X)    
    return [X, W]

def GaussLobattoJacobiQuadrature1D(num_points, a, b, alpha = 0, beta = 0):

    roots, weights = GaussLobattoJacobiWeights(num_points, alpha, beta)
    X = (b-a)/2 * (roots+1) + a

    return X, weights

def GaussLobattoJacobiQuadrature2D(num_points_per_dim, a, b, c, d,
                                   alpha_x = 0, beta_x = 0, alpha_y = 0, beta_y = 0):
    '''
    Returns the num_points points and weights of the Gauss-Lobatto-Jacobi quadrature over [a,b]*[c,d].
    This method is accurate for polynomials up to degree 2num_points-3 in 1D.
    '''

    roots_X, weights_X = GaussLobattoJacobiQuadrature1D(num_points_per_dim, a, b, alpha_x, beta_x)
    roots_Y, weights_Y = GaussLobattoJacobiQuadrature1D(num_points_per_dim, c, d, alpha_y, beta_y)

    X, Y = np.meshgrid(roots_X, roots_Y)
    weights_X, weights_Y = np.meshgrid(weights_X, weights_Y)

    return X, Y, weights_X, weights_Y

# INTEGRATION ALGORITHMS
##################################################

def Integrate1D(f, a, b, X = None, weights = None):
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

    if type(weights).__name__ != "ndarray" and type(weights).__name__ != "Tensor":
        weights = torch.Tensor(weights)

    if X is not None and type(X).__name__ != "ndarray" and type(X).__name__ != "Tensor":
        X = torch.Tensor(X)
        
    if type(f).__name__ == 'function':
        if X is None:
            raise ValueError("f is a function handle but no values were provided for evaluation (roots = None)")
        values = f(X)
    elif type(f).__name__ != "ndarray" and type(f).__name__ != "Tensor":
        values = torch.Tensor(f)
    else:
        values = f
    
    if  type(values).__name__ == 'Tensor':
        integral = (b-a) * torch.sum(values*weights) / 2
    else:
        integral = (b-a) * np.sum(values*weights) / 2
    
    return integral

def Integrate2D(f, a, b, c, d, X = None,
                weights_X = None, Y = None,
                weights_Y = None):
    '''
    Integrates a function f over [a, b]*[c,d] using the given roots and weights
    Parameters:
        f (iterable or function handle): if f is an array of the same size as weights,
                                         the integral is directly computed using summation.
                                         Otherwise f is evaluated on the roots before hand.
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
    if X is not None and type(X).__name__ != "ndarray" and type(X).__name__ != "Tensor":
        X = torch.Tensor(X)
    if Y is not None and type(Y).__name__ != "ndarray" and type(Y).__name != "Tensor":
        Y = torch.Tensor(Y)
    if type(weights_X).__name__ != "ndarray" and type(weights_X).__name__ != "Tensor":
        weights_X = torch.Tensor(weights_X)
    if type(weights_Y).__name__ != "ndarray" and type(weights_Y).__name__ != "Tensor":
        weights_Y = torch.Tensor(weights_Y)
        
    if type(f).__name__ == 'function':
        if X is None or Y is None:
            raise ValueError("f is a function handle but no values were provided for evaluation (roots = None)")
        values = f(X, Y)
    elif type(f).__name__ != "ndarray" and type(f).__name__ != "Tensor":
        values = torch.Tensor(f)
    else:
        values = f

    if type(values).__name__ == 'Tensor':
        integral = (b-a) * (d-c) * torch.sum(values * weights_X * weights_Y) / 4
    else:
        integral = (b-a) * (d-c) * np.sum(values * weights_X * weights_Y) / 4

    return integral