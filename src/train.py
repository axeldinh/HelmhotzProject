from tqdm import tqdm

def train(vpinn, optimizer, num_iter):
    t = tqdm(range(num_iter))
    last_loss = 0
    for iter_ in t:
        optimizer.zero_grad()
        loss_interior, loss_boundary = vpinn.compute_loss(3)
        loss = loss_interior + loss_boundary
        loss.backward(retain_graph=True)
        for n, p in vpinn.model.named_parameters():
            vpinn.grad_parameters[n].append(p.grad.clone())
        optimizer.step()
        t.set_description("Loss Interior = {:.7f}, Loss Boundary = {:.7f}".format(loss_interior.item(), loss_boundary.item()))
        if torch.abs(last_loss - loss) < 1e-10:
            break
        last_loss = loss
    print("Training Over")