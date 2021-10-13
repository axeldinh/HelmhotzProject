import sys
sys.path.append("./src")

import torch_derivatives

import unittest
import torch

class Test_compute_derivatives(unittest.TestCase):

    def test_square(self):
        '''
        Test the compute_derivative(u,x) function with u = x**2
        '''
    
        x = torch.arange(3).view(-1,1).float()
        x.requires_grad = True
        u = x**2
        u_x = torch_derivatives.compute_derivative(u, x)
        expected = 2*x.clone()
        self.assertEqual(torch.all(torch.eq(u_x, expected)), True, msg=f"\n\nu_x should be \n{expected} \nbut got\n {u_x}")

    def test_cube(self):
        '''
        Test the compute_derivative(u,x) function with u = x1**3 + x2**3 + x3**3
        '''
    
        x = torch.arange(12).view(4,3).float()
        x.requires_grad = True
        u = torch.sum(x**3, dim = 1)
        u_x = torch_derivatives.compute_derivative(u, x)
        expected = (3*x**2).clone()
        self.assertEqual(torch.all(torch.eq(u_x, expected)), True, msg=f"\n\nu_x should be \n{expected} \nbut got\n {u_x}")

    def test_exp(self):
        '''
        Test the compute_derivative(u,x) function with u = exp(x1) + exp(x2) + exp(x3)
        '''
    
        x = torch.arange(12).view(4,3).float()
        x.requires_grad = True
        u = torch.sum(torch.exp(x), dim = 1)
        u_x = torch_derivatives.compute_derivative(u, x)
        expected = torch.exp(x).clone()
        self.assertEqual(torch.all(torch.eq(u_x, expected)), True, msg=f"\n\nu_x should be \n{expected} \nbut got\n {u_x}")

    def test_mult(self):
        '''
        Test the compute_derivative(u,x) function with u = x1*x2
        '''
    
        x = torch.arange(8).view(4,2).float()
        x.requires_grad = True
        u = x[:,0]*x[:,1]
        u_x = torch_derivatives.compute_derivative(u, x)
        expected = x[:, [1,0]].clone()
        self.assertEqual(torch.all(torch.eq(u_x, expected)), True, msg=f"\n\nu_x should be \n{expected} \nbut got\n {u_x}")

class Test_compute_laplacian(unittest.TestCase):

    def test_square(self):
        '''
        Test the compute_laplacian(u,x) function with u = x1**2 + x2**2 + x3**2
        '''

        x = torch.arange(4).view(-1,1).float()
        y = torch.arange(4,8).view(-1,1).float()
        z = torch.arange(8,12).view(-1,1).float()
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True

        u = x**2 + y**2 + z**2
        u_lapl = torch_derivatives.compute_laplacian(u, [x, y, z])

        expected = torch.sum(2*torch.ones((4, 3)), dim = 1)
        self.assertEqual(torch.all(torch.eq(u_lapl, expected)), True, msg=f"\n\nu_lapl should be \n{expected} \nbut got\n {u_lapl}")

    def test_cube(self):
        '''
        Test the compute_laplacian(u,x) function with u = x1**3 + x2**3 + x3**3
        '''

        x = torch.arange(4).view(-1,1).float()
        y = torch.arange(4,8).view(-1,1).float()
        z = torch.arange(8,12).view(-1,1).float()
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True

        u = x**3 + y**3 + z**3
        u_lapl = torch_derivatives.compute_laplacian(u, [x, y, z])

        expected = torch.sum(torch.cat([6*x, 6*y, 6*z], dim = 1), dim = 1).clone()
        self.assertEqual(torch.all(torch.eq(u_lapl, expected)), True, msg=f"\n\nu_lapl should be \n{expected} \nbut got\n {u_lapl}")

    def test_exp(self):
        '''
        Test the compute_laplacian(u,x) function with u = exp(x1) + exp(x2) + exp(x3)
        '''

        x = torch.arange(4).view(-1,1).float()
        y = torch.arange(4,8).view(-1,1).float()
        z = torch.arange(8,12).view(-1,1).float()
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True

        u = torch.exp(x) + torch.exp(y) + torch.exp(z)
        u_lapl = torch_derivatives.compute_laplacian(u, [x, y, z])

        expected = u.clone().view(u_lapl.size())
        self.assertEqual(torch.all(torch.eq(u_lapl, expected)), True, msg=f"\n\nu_lapl should be \n{expected} \nbut got\n {u_lapl}")

    def test_mult(self):
        '''
        Test the compute_lalacian(u,x) function with u = x1*x2
        '''

        x = torch.arange(4).view(-1,1).float()
        y = torch.arange(4,8).view(-1,1).float()
        z = torch.arange(8,12).view(-1,1).float()
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True

        u = x * y * z
        u_lapl = torch_derivatives.compute_laplacian(u, [x, y, z])

        expected = torch.zeros((4,)).view(u_lapl.size())
        self.assertEqual(torch.all(torch.eq(u_lapl, expected)), True, msg=f"\n\nu_lapl should be \n{expected} \nbut got\n {u_lapl}")

if __name__=="__main__":

    unittest.main()


