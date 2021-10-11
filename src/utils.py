from scipy.special import jacobi
import numpy as np
import torch
from torch._C import Value

def getBack(var_grad_fn):
    '''
    Tracks the autograd tree.
    Parameters:
        var_grad_fn: variable recovered with torch.Tensor().grad_fn
    '''
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])

def getPolyTest(k, x):

    if type(x).__name__ != 'Tensor':
        raise ValueError(f"x should be a Tensor, currently is of type {type(x)}")

    poly1 = jacobi(k-1, 0, 0)
    poly2 = jacobi(k+1, 0, 0)
    poly1 = list(poly1)[::-1]
    poly2 = list(poly2)[::-1]
    poly = torch.zeros(x.size(), dtype = torch.double, requires_grad=True).to(x.device)

    for i in range(len(poly2)):
        poly = poly + poly2[i]*x**i
        if i < len(poly1):
            poly = poly - poly1[i]*x**i

    return poly