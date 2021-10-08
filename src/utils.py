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