import sys
sys.path.append('./src')

from VPINN import *
import torch
import numpy as np

import unittest

class Test_VPINN_Laplace(unittest.TestCase):

    def testExactSolutionGivesZeroLoss(self):
        '''
        Check if the loss is 0 when starting at the exact solution
        '''

        a, b = -1, 1
        A, w = 1., 2.1*np.pi
        u_ex = lambda x: A*torch.sin(w*x)
        u_left = A*torch.sin(torch.tensor([a*w], dtype = torch.double))
        u_right = A*torch.sin(torch.tensor([b*w], dtype = torch.double))
        source_function = lambda x: A*w**2 * torch.sin(w*x)

        num_points = 80
        num_sine_test_functions = 5
        num_poly_test_functions = 5
        boundary_penalty = 5

        vpinn = VPINN_Laplace_Dirichlet(a, b, u_left, u_right, source_function,
                 num_points, num_sine_test_functions, num_poly_test_functions,
                 boundary_penalty, layers=None, activation=None, datas = None,
                 u_handle = u_ex, u_ex = u_ex, device = 'cpu')

        loss_interior, loss_boundary = vpinn.compute_loss(1)
        loss1 = loss_interior + loss_boundary
        self.assertAlmostEqual(loss1.item(), 0.0, msg="For method 1 VPINN_Laplace_Dirichlet does not give 0 loss with the exact solution")
        loss_interior, loss_boundary = vpinn.compute_loss(2)
        loss2 = loss_interior + loss_boundary
        self.assertAlmostEqual(loss2.item(), 0.0, msg="For method 2 VPINN_Laplace_Dirichlet does not give 0 loss with the exact solution")
        loss_interior, loss_boundary = vpinn.compute_loss(3)
        loss3 = loss_interior + loss_boundary
        self.assertAlmostEqual(loss3.item(), 0.0, msg="For method 3 VPINN_Laplace_Dirichlet does not give 0 loss with the exact solution")

class Test_VPINN_SteadyBurger(unittest.TestCase):

    def testExactSolutionGivesZeroLoss(self):
        '''
        Check if the loss is 0 when starting at the exact solution
        '''

        # Check VPINN_SteadyBurger_Dirichlet
        ##################################################

        a, b = -1, 1
        A, w = 1., 2.1*np.pi
        u_ex = lambda x: A*torch.sin(w*x)
        u_left = A*torch.sin(torch.tensor([a*w], dtype = torch.double))
        u_right = A*torch.sin(torch.tensor([b*w], dtype = torch.double))
        source_function = lambda x: 0.5 * A**2 * w * torch.sin(2*w*x) + A * w**2 * torch.sin(w*x)

        num_points = 80
        num_sine_test_functions = 5
        num_poly_test_functions = 5
        boundary_penalty = 5

        vpinn = VPINN_SteadyBurger_Dirichlet(a, b, u_left, u_right, source_function,
                 num_points, num_sine_test_functions, num_poly_test_functions,
                 boundary_penalty, layers=None, activation=None, datas = None,
                 u_handle = u_ex, u_ex = u_ex, device = 'cpu')

        loss_interior, loss_boundary = vpinn.compute_loss(1)
        loss1 = loss_interior + loss_boundary
        self.assertAlmostEqual(loss1.item(), 0.0, msg="For method 1 VPINN_SteadyBurger_Dirichlet does not give 0 loss with the exact solution")
        loss_interior, loss_boundary = vpinn.compute_loss(2)
        loss2 = loss_interior + loss_boundary
        self.assertAlmostEqual(loss2.item(), 0.0, msg="For method 2 VPINN_SteadyBurger_Dirichlet does not give 0 loss with the exact solution")
        loss_interior, loss_boundary = vpinn.compute_loss(3)
        loss3 = loss_interior + loss_boundary
        self.assertAlmostEqual(loss3.item(), 0.0, msg="For method 3 VPINN_SteadyBurger_Dirichlet does not give 0 loss with the exact solution")

if __name__ == "__main__":

    unittest.main()