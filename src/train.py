import numpy as np
from tqdm import tqdm
import torch
import copy

def train(model, optimizer, num_iter, method = 1, tol = 1e-10, save_gradients = False):
    """
    Runs the model num_iter times, it ends if the loss stops converging (with a tolerance of tol)
    During the process, the best model is saved and fed into the model at the the end of the training.
    """
    t = tqdm(range(num_iter))
    last_loss = 0
    best_loss = np.inf

    for iter_ in t:
        if optimizer is not None:
            optimizer.zero_grad()
        loss_interior, loss_boundary = model.compute_loss(method)
        loss = loss_interior + loss_boundary
        loss.backward(retain_graph=True)

        if loss < best_loss:
            best_loss = loss
            model.best_model = copy.deepcopy(model.model)

        # If we want we can save the gradients
        if type(model.model).__name__ != "function" and save_gradients:
            for n, p in model.model.named_parameters():
                model.grad_parameters[n].append(p.grad.clone())

        if optimizer is not None:
            optimizer.step()
        t.set_description("Loss Interior = {:e}, Loss Boundary = {:e}".format(loss_interior.item(), loss_boundary.item()))
        if torch.abs(last_loss - loss) < tol:
            break
        last_loss = loss
    
    model.model = copy.deepcopy(model.best_model)

    print("Training Over")