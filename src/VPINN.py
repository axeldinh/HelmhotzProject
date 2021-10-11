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
                 u_handle = None, u_ex = None, device = 'cpu'):
        
        super().__init__()
        
        self.a, self.b = a, b
        self.num_points = num_points
        self.num_sine_test_functions = num_sine_test_functions
        self.num_poly_test_functions = num_poly_test_functions
        self.num_test_functions = num_sine_test_functions+num_poly_test_functions
        self.u_left, self.u_right = u_left.to(device), u_right.to(device)
        self.boundary_penalty = boundary_penalty
        self.source_function = source_function
        
        # Set the quadrature points and weights
        self.x, self.weights = qr.GaussLobattoJacobiQuadrature1D(num_points, a, b)
        self.x = torch.tensor(self.x, dtype = torch.double).view(-1,1).to(device)
        self.weights = torch.Tensor(self.weights).double().to(device)
        self.x.requires_grad = True
        
        # Compute the test function with their derivatives
        self.test_functions = torch.zeros((self.num_test_functions, num_points)).to(device)
        self.dtest_functions = torch.zeros((self.num_test_functions, num_points)).to(device)
        self.d2test_functions = torch.zeros((self.num_test_functions, num_points)).to(device)

        # Sine functions
        for k in range(1, num_sine_test_functions+1):
            self.test_functions[k-1] = torch.sin(np.pi*k*self.x.view(-1)).to(device)
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
            
        self.source = source_function(self.x).view(-1).to(device)
        
        if u_handle is not None:
            self.model = u_handle
        else:
            self.model = MLP(layers, activation, True, datas).to(device)
        self.u = self.model(self.x).to(device)
        self.u_ex = u_ex
        
        self.losses_interior = []
        self.losses_boundary = []

        if type(self.model).__name__ != "function":
            self.grad_parameters = {}
            for n, _ in self.model.named_parameters():
                self.grad_parameters[n] = []

    def IntegrateOnTest(self, f, degree_derivative=0):
        '''
        Computes the integral of f*test_function for all test functions,
        degree_derivative specifies which derivative of test_function is needed
        '''
        if degree_derivative==0:
          integrals = self.weights * f * self.test_functions
        elif degree_derivative==1:
          integrals = self.weights * f * self.dtest_functions
        elif degree_derivative==2:
          integrals = self.weights * f * self.d2test_functions

        return integrals.sum(dim = 1)
    
    def compute_F(self):
        '''
        Computes integral(f * vk) for all k
        '''
        return self.IntegrateOnTest(self.source)
    
    def compute_R_NL(self, u):
        '''
        Computes integral(u * du/dx * vk) for all k
        '''
        u_x = td.compute_derivative(u, self.x, retain_graph=True, create_graph=True).view(-1)
        return self.IntegrateOnTest(u.view(-1)*u_x)

    def compute_R1(self, u):
        '''
        Computes -integral(d^2u/dx^2 * vk) for all k
        '''
        u_xx = td.compute_laplacian(u, [self.x], retain_graph=True, create_graph=True).view(-1)
        return self.IntegrateOnTest(-u_xx)

    def compute_R2(self, u):
        '''
        Computes integral(du/dx * dvk/dx) for all k
        '''
        u_x = td.compute_derivative(u, self.x, retain_graph=True, create_graph=True).view(-1)
        return self.IntegrateOnTest(u_x, 1)

    def compute_R3(self, u, k):
        '''
        Computes integral(u * d^2vk/dx^2) + Integral_boundary(u * dvk/dx) for all k
        '''
        integral = -self.IntegrateOnTest(u.view(-1), 2)
        BC = self.u_right * self.dtest_functions[:,-1] -self.u_left*self.dtest_functions[:,0]
        return integral + BC
    
    def compute_R(self):
        raise NotImplementedError
    
    def compute_loss(self, method = 1):
        
        self.u = self.model(self.x)

        R = self.compute_R(self.u, method = method)
        F = self.compute_F()
        loss_interior = torch.mean((R-F)**2)
        self.losses_interior.append(loss_interior.item())
            
        loss_boundary = (self.u[0] - self.u_left)**2 + (self.u[-1] - self.u_right)**2
        loss_boundary *= self.boundary_penalty / 2
        self.losses_boundary.append(loss_boundary.item())
        
        return loss_interior, loss_boundary

    def plot(self):
        if self.u_ex:
            plt.plot(self.x.cpu().detach().numpy(), self.u_ex(self.x).cpu().detach().numpy(), label="Exact", c = 'r')
        plt.scatter(self.x.cpu().detach().numpy(), self.u.cpu().detach().numpy(), label="Approx", c = 'k', marker = "*", s = 20)
        plt.grid()
        plt.legend()
        plt.show()

class VPINN_Laplace_Dirichlet(VPINN):
    
    def __init__(self, a, b, u_left, u_right, source_function,
                 num_points, num_test_functions, num_poly_test_functions,
                 boundary_penalty, layers, activation, datas = None,
                 u_handle = None, u_ex = None, device = 'cpu'):

        super().__init__(a, b, u_left, u_right, source_function,
                 num_points, num_test_functions, num_poly_test_functions,
                 boundary_penalty, layers, activation, datas,
                 u_handle, u_ex, device = device)

    def compute_Rk(self, u, k, method = 1):
            
        if method == 1: Rk = self.compute_Rk1(u, k)
        elif method == 2: Rk = self.compute_Rk2(u, k)
        elif method == 3: Rk = self.compute_Rk3(u, k)
            
        return Rk
    
    def compute_R(self, u, method = 1):
      
        if method == 1: R = self.compute_R1(u)
        elif method == 2: R = self.compute_R2(u)
        elif method == 3: R = self.compute_R3(u)

        return R


class VPINN_SteadyBurger_Dirichlet(VPINN):
    
    def __init__(self, a, b, u_left, u_right, source_function,
                 num_points, num_test_functions, num_poly_test_functions,
                 boundary_penalty, layers, activation, datas = None,
                 u_handle = None, u_ex = None, device = 'cpu'):
        
        super().__init__(a, b, u_left, u_right, source_function,
                 num_points, num_test_functions, num_poly_test_functions,
                 boundary_penalty, layers, activation, datas,
                 u_handle, u_ex, device = device)

    def compute_Rk(self, u, k, method = 1):
            
        if method == 1: Rk = self.compute_Rk1(u, k)
        elif method == 2: Rk = self.compute_Rk2(u, k)
        elif method == 3: Rk = self.compute_Rk3(u, k)
            
        Rk += self.compute_Rk_NL(u, k)
        
        return Rk
    
    def compute_R(self, u, method = 1):
      
        if method == 1: R = self.compute_R1(u)
        elif method == 2: R = self.compute_R2(u)
        elif method == 3: R = self.compute_R3(u)

        R += self.compute_R_NL(u)

        return R


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