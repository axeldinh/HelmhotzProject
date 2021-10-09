from tqdm import tqdm
import torch

def train(model, optimizer, num_iter, method = 1):
    t = tqdm(range(num_iter))
    last_loss = 0
    for iter_ in t:
        if optimizer is not None:
            optimizer.zero_grad()
        loss_interior, loss_boundary = model.compute_loss(method)
        loss = loss_interior + loss_boundary
        loss.backward(retain_graph=True)
        if type(model.model).__name__ != "function":
            for n, p in model.model.named_parameters():
                model.grad_parameters[n].append(p.grad.clone())
        if optimizer is not None:
            optimizer.step()
        t.set_description("Loss Interior = {:.7f}, Loss Boundary = {:.7f}".format(loss_interior.item(), loss_boundary.item()))
        if torch.abs(last_loss - loss) < 1e-10:
            break
        last_loss = loss
    print("Training Over")