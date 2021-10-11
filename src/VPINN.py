import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

import torch_derivatives as td
import quadrature_rules as qr
from utils import *

class MLP(nn.Module):
    
    def __init__(self, layers, activation, bias = True, datas = None):
        
        super().__init__()
        
        self.activation = activation
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1], bias = bias, dtype = torch.double)
                                      for i in range(len(layers)-1)])
        
        if datas:
            for data, lin in zip(datas, self.linears):
                lin.weight.data = data['weight']
                if lin.bias is not None:
                    lin.bias.data = data['bias']
        
    def forward(self, x):
        
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears)-1:
                x = self.activation(x)
                
        return x

# VPINNs
##################################################

class VPINN(nn.Module):
    
    def __init__(self, a, b, u_left, u_right, source_function,
                 num_points, num_sine_test_functions, num_poly_test_functions,
                 boundary_penalty, layers, activation, datas = None,
                 u_handle = None, u_ex = None):
        
        super().__init__()
        
        self.a, self.b = a, b
        self.num_points = num_points
        self.num_sine_test_functions = num_sine_test_functions
        self.num_poly_test_functions = num_poly_test_functions
        self.num_test_functions = num_sine_test_functions+num_poly_test_functions
        self.u_left, self.u_right = u_left, u_right
        self.boundary_penalty = boundary_penalty
        self.source_function = source_function
        
        # Set the quadrature points and weights
        self.x, self.weights = qr.GaussLobattoJacobiQuadrature1D(num_points, a, b)
        self.x = torch.tensor(self.x, dtype = torch.double).view(-1,1)
        self.weights = torch.Tensor(self.weights).double()
        self.x.requires_grad = True
        
        # Compute the test function with their derivatives
        self.test_functions = torch.zeros((self.num_test_functions, num_points))
        self.dtest_functions = torch.zeros((self.num_test_functions, num_points))
        self.d2test_functions = torch.zeros((self.num_test_functions, num_points))

        # Sine functions
        for k in range(1, num_sine_test_functions+1):
            self.test_functions[k-1] = torch.sin(np.pi*k*self.x.view(-1))
            self.dtest_functions[k-1] = td.compute_derivative(self.test_functions[k-1],
                                                              self.x, retain_graph = True).view(-1)
            self.d2test_functions[k-1] = td.compute_laplacian(self.test_functions[k-1], [self.x], retain_graph=True).view(-1)

        # Polynomials
        for k in range(1, num_poly_test_functions+1):
            ind = k-1+num_sine_test_functions
            poly = getPolyTest(k, self.x).view(-1)
            self.test_functions[ind] = poly
            self.dtest_functions[ind] = td.compute_derivative(self.test_functions[ind],
                                                              self.x, retain_graph = True).view(-1)
            self.d2test_functions[ind] = td.compute_laplacian(self.test_functions[ind], [self.x], retain_graph=True).view(-1)
            
        self.source = source_function(self.x).view(-1)
        
        if u_handle is not None:
            self.model = u_handle
        else:
            self.model = MLP(layers, activation, True, datas)
        self.u = self.model(self.x)
        self.u_ex = u_ex
        
        self.losses_interior = []
        self.losses_boundary = []

        if type(self.model).__name__ != "function":
            self.grad_parameters = {}
            for n, _ in self.model.named_parameters():
                self.grad_parameters[n] = []
        
    def Integrate(self, f):
        return qr.Integrate1D(f, self.a, self.b, self.x, self.weights)
        
    def compute_Fk(self, k):
        '''
        Computes integral(f * vk)
        '''
        return self.Integrate(self.source * self.test_functions[k])
    
    def compute_Rk_NL(self, u, k):
        '''
        Computes integral(u * du/dx * vk)
        '''
        u_x = td.compute_derivative(u, self.x, retain_graph=True, create_graph=True).view(-1)
        return self.Integrate(u.view(-1)*u_x*self.test_functions[k])
    
    def compute_Rk1(self, u, k):
        '''
        Computes integral(d^2u/dx^2 * vk)
        '''
        u_xx = td.compute_laplacian(u, [self.x], retain_graph=True, create_graph=True).view(-1)
        return -self.Integrate(u_xx * self.test_functions[k])
    
    def compute_Rk2(self, u, k):
        '''
        Computes integral(du/dx * dvk/dx)
        '''
        u_x = td.compute_derivative(u, self.x, retain_graph=True, create_graph=True).view(-1)
        return self.Integrate(u_x * self.dtest_functions[k])
    
    def compute_Rk3(self, u, k):
        '''
        Computes integral(u * d^2vk/dx^2) + Integral_boundary(u * dvk/dx)
        '''
        return -self.Integrate(u.view(-1) * self.d2test_functions[k]) + \
                self.u_right*self.dtest_functions[k][-1] - self.u_left*self.dtest_functions[k][0]
    
    def compute_Rk(self):
        raise NotImplementedError
    
    def compute_loss(self, method = 1):
        
        self.u = self.model(self.x)
        loss_interior = 0
        for k in range(self.num_test_functions):
            Rk = self.compute_Rk(self.u, k, method)
            Fk = self.compute_Fk(k)
            loss_interior += (Rk - Fk)**2
        loss_interior /= self.num_test_functions
        self.losses_interior.append(loss_interior.item())
            
        loss_boundary = (self.u[0] - self.u_left)**2 + (self.u[-1] - self.u_right)**2
        loss_boundary *= self.boundary_penalty / 2
        self.losses_boundary.append(loss_boundary.item())
        
        return loss_interior, loss_boundary

    def plot(self):
        if self.u_ex:
            plt.plot(self.x.detach().numpy(), self.u_ex(self.x).detach().numpy(), label="Exact", c = 'r')
        plt.scatter(self.x.detach().numpy(), self.u.detach().numpy(), label="Approx", c = 'k', marker = "*", s = 20)
        plt.grid()
        plt.legend()
        plt.show()

class VPINN_Laplace_Dirichlet(VPINN):
    
    def __init__(self, a, b, u_left, u_right, source_function,
                 num_points, num_test_functions, num_poly_test_functions,
                 boundary_penalty, layers, activation, datas = None,
                 u_handle = None, u_ex = None):

        super().__init__(a, b, u_left, u_right, source_function,
                 num_points, num_test_functions, num_poly_test_functions,
                 boundary_penalty, layers, activation, datas,
                 u_handle, u_ex)

    def compute_Rk(self, u, k, method = 1):
            
        if method == 1: Rk = self.compute_Rk1(u, k)
        elif method == 2: Rk = self.compute_Rk2(u, k)
        elif method == 3: Rk = self.compute_Rk3(u, k)
            
        return Rk

class VPINN_SteadyBurger_Dirichlet(VPINN):
    
    def __init__(self, a, b, u_left, u_right, source_function,
                 num_points, num_test_functions, num_poly_test_functions,
                 boundary_penalty, layers, activation, datas = None,
                 u_handle = None, u_ex = None):
        
        super().__init__(a, b, u_left, u_right, source_function,
                 num_points, num_test_functions, num_poly_test_functions,
                 boundary_penalty, layers, activation, datas,
                 u_handle, u_ex)

    def compute_Rk(self, u, k, method = 1):
            
        if method == 1: Rk = self.compute_Rk1(u, k)
        elif method == 2: Rk = self.compute_Rk2(u, k)
        elif method == 3: Rk = self.compute_Rk3(u, k)
            
        Rk += self.compute_Rk_NL(u, k)
        
        return Rk


# PINNs
##################################################

class PINN(nn.Module):
    
    def __init__(self, a, b, u_left, u_right, source_function,
                 num_points, boundary_penalty,
                 layers, activation, u_ex = None):
        
        super().__init__()
        
        self.a, self.b = a, b
        self.num_points = num_points
        self.u_left, self.u_right = u_left, u_right
        self.boundary_penalty = boundary_penalty
        self.source_function = source_function
        
        # Set the quadrature points
        self.x, _ = qr.GaussLobattoJacobiQuadrature1D(num_points, a, b)
        self.x = torch.tensor(self.x, dtype = torch.double).view(-1,1)
        self.x.requires_grad = True
            
        self.source = source_function(self.x).view(-1)
        
        if u_ex is not None:
            self.model = u_ex
        else:
            self.model = MLP(layers, activation)
        self.u = None
        
        self.losses_interior = []
        self.losses_boundary = []
        self.grad_parameters = {}

        if type(self.model).__name__ != "function":
            for n, p in self.model.named_parameters():
                self.grad_parameters[n] = []
    
    def compute_R(self, u):
        raise NotImplementedError
    
    def compute_loss(self, method = 1):
        
        self.u = self.model(self.x)
        loss_interior = (self.compute_R(self.u) - self.source)**2
        loss_interior = torch.mean(loss_interior)
        self.losses_interior.append(loss_interior.item())
            
        loss_boundary = (self.u[0] - self.u_left)**2 + (self.u[-1] - self.u_right)**2
        loss_boundary *= self.boundary_penalty / 2
        self.losses_boundary.append(loss_boundary.item())
        
        return loss_interior, loss_boundary

class PINN_SteadyBurger_Dirichlet(PINN):
    
    def __init__(self, a, b, u_left, u_right, source_function,
                 num_points, boundary_penalty,
                 layers, activation, u_ex = None):
        
        super().__init__(a, b, u_left, u_right, source_function,
                 num_points, boundary_penalty,
                 layers, activation, u_ex)
    
    def compute_R(self, u):
        
        u_x = td.compute_derivative(u, self.x, retain_graph=True, create_graph=True).view(-1)
        u_xx = td.compute_laplacian(u, [self.x], retain_graph = True, create_graph=True).view(-1)
        
        return u.view(-1)*u_x - u_xx
    
class PINN_Laplace_Dirichlet(PINN):
    
    def __init__(self, a, b, u_left, u_right, source_function,
                 num_points, boundary_penalty,
                 layers, activation, u_ex = None):
        
        super().__init__(a, b, u_left, u_right, source_function,
                 num_points, boundary_penalty,
                 layers, activation, u_ex)
    
    def compute_R(self, u):
        
        u_xx = td.compute_laplacian(u, [self.x], retain_graph = True).view(-1)
        
        return - u_xx