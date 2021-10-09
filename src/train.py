from tqdm import tqdm
import torch

def train(model, optimizer, num_iter, method = 1):
    t = tqdm(range(num_iter))
    last_loss = 0
    for iter_ in t:
        optimizer.zero_grad()
        loss_interior, loss_boundary = model.compute_loss(method)
        loss = loss_interior + loss_boundary
        loss.backward(retain_graph=True)
        if model.grad_parameters is not None:
            for n, p in model.model.named_parameters():
                model.grad_parameters[n].append(p.grad.clone())
        optimizer.step()
        t.set_description("Loss Interior = {:.7f}, Loss Boundary = {:.7f}".format(loss_interior.item(), loss_boundary.item()))
        if torch.abs(last_loss - loss) < 1e-10:
            break
        last_loss = loss
    print("Training Over")