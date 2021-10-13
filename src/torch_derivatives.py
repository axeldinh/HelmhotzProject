import torch

def compute_derivative(u, x, retain_graph = False, create_graph = False, allow_unused = False):
    '''
    Computes the derivative of u with respect to x. 
    E.g.: If u = [u1, u2, u3], and x = [x1, x2, x3],
    the function will return u_x = [du1/dx(x1), du2/dx(x2), du3/dx(x3), du3/dx(x3)]
    
        Parameters:
            u (torch.Tensor): Tensor of size (x.size(0)), it corresponds to the function needed to be derived
            x (torch.Tensor): Tensor of size (number of samples, D), where D is the dimension of the problem.
                            x should require grad (x.requires_grad = True).
            retain_graph (Bool): Parameter for the torch.autograd.backward function
            create_graph (Bool): Parameter for the torch.autograd.backward function
        
        Example:
            x = torch.arange(12).view(4,3).float()
            x.requires_grad = True
            u = torch.sum(x**2, dim = 1)
            u_x = compute_derivative(u, x)
    '''

    # CHECKS
    #########################################
    if not x.requires_grad:
        raise ValueError("x.requires_grad is set to False")
        
    if len(x.size()) > 2 and x.size(1) > 1:
        raise ValueError(f"x is a {len(x.size())}D Tensor, please choose a 1D Tensor. x has size {x.size()}")

    if len(u.size()) > 2 and u.size(1) > 1:
        raise ValueError(f"u is a {len(u.size())}D Tensor, please choose a 2D Tensor (u should return a scalar per point). u has size {u.size()}")

    if len(u.size()) > 1:
        if u.size(1) > 1:
            raise ValueError(f"More than one value is associated to each point, u should either be a 1D Tensor or a flat 2D Tensor,  u has size {u.size()}")

    grad = torch.ones(u.size()).to(x.device)
    u_x = torch.autograd.grad([u], inputs = x, grad_outputs=[grad],
                              retain_graph=retain_graph, create_graph=create_graph, allow_unused=allow_unused)[0]
    
    # In case x was unused (e.g for 2nd derivative of u = x), we return the zero vector
    if u_x==None:
        u_x = torch.zeros(x.size())
    if x.grad is not None:
        x.grad.zero_()
    
    return u_x

def compute_laplacian(u, inputs, retain_graph = False, create_graph = False):
    '''
    Computes the Laplacian of u with respect to x. 
    E.g.: If u = [u1, u2, u3], and x = [[x11, x12], [x21, x22]],
    the function will return u_x = [d^2u1/dx11^2 + d^2u1/dx12^2, d^2u2/dx21^2 + d^2u2/dx22^2]
    
        Parameters:
            u (torch.Tensor): Tensor of size (x.size(0)), it corresponds to the function needed to be derived
            inputs (sequence of torch.Tensor): Sequence of D Tensors of size (number of samples,), 
                                               where D is the dimension of the problem.
                                               Each Tensor should require grad (x.requires_grad = True).
            retain_graph (Bool): Parameter for the torch.autograd.backward function
            create_graph (Bool): Parameter for the torch.autograd.backward function
                            
    Example:
        x = torch.arange(4).view(-1,1).float()
        y = torch.arange(4).view(-1,1).float()
        x.requires_grad = True
        y.requires_grad = True
        u = x**2 + y**2
        u_lapl = compute_laplacian(u, [x,y])
    '''

    # CHECKS
    #########################################

    if type(inputs).__name__ != 'list' and type(inputs).__name__ != 'tuple':
        raise ValueError("Inputs should be a list or a tuple.")

    for x in inputs:
        if not x.requires_grad:
            raise ValueError(f"x.requires_grad is set to False, got {[a.requires_grad for a in inputs]}")
            
        if len(x.size()) > 2 and x.size(1) > 1:
            raise ValueError(f"x is a {len(x.size())}D Tensor, please choose a 1D Tensor. x has size {x.size()}")

    if len(u.size()) > 2:
        raise ValueError(f"u is a {len(u.size())}D Tensor, please choose a 2D Tensor (u should return a scalar per point). u has size {u.size()}")

    if len(u.size()) > 1:
        if u.size(1) > 1:
            raise ValueError(f"More than one value is associated to each point, u should either be a 1D Tensor or a flat 2D Tensor,  u has size {u.size()}")
        
    lapl = []

    for x in inputs:
        u_clone = u.clone()
        u_x = compute_derivative(u_clone, x, retain_graph=True, create_graph=True)
        u_xx = compute_derivative(u_x, x, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True)
        lapl.append(u_xx.view(-1, 1))
        
    return torch.sum(torch.cat(lapl, dim = 1), dim = 1)