
import copy
import torch.nn as nn
import torch.optim as optim
import torch.func as stateless
import GenerateData as gd
import torch


class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FTheta(nn.Module):
    def __init__(self, dx: int = 1, dy: int = 1):
        super().__init__()
        self.model = MLP([dx + dy, 8, 8, dy])

    def forward(self, x_in):
        return self.model(x_in)


def compute_predictions(model, X, k):
    T = X.shape[0]
    y = torch.zeros(T)
    y_prev = torch.tensor(0.0)

    for t in range(k, T):
        inp = torch.tensor([X[t - k].item(), y_prev.item()], dtype=X.dtype)
        y_t = model(inp)
        y[t] = y_t
        y_prev = y_t.detach()

    return y


def omd_step(model, loss_fn, X, k, eta):
    """
    Performs OMD:

        w_wait = w - eta * grad L(w)
        w_next = w - eta * grad L(w_wait)
    """

    # -------------------------------
    # First gradient: grad L(w)
    # -------------------------------
    y_pred = compute_predictions(model, X, k)
    loss = loss_fn(y_pred[k:], X[k:])
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

    # -------------------------------
    # Build w_wait model by cloning
    # -------------------------------
    model_wait = copy.deepcopy(model)
    with torch.no_grad():
        for (name, p_wait), (_, p_orig), g in zip(
            model_wait.named_parameters(),
            model.named_parameters(),
            grads,
        ):
            p_wait -= eta * g

    # -------------------------------
    # Second gradient: grad L(w_wait)
    # -------------------------------
    y_pred_wait = compute_predictions(model_wait, X, k)
    loss_wait = loss_fn(y_pred_wait[k:], X[k:])
    grads_wait = torch.autograd.grad(loss_wait, model_wait.parameters())

    # -------------------------------
    # Final update: w_next = w - eta * grad L(w_wait)
    # -------------------------------
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads_wait):
            p -= eta * g



T = 1000
k = 1

Xsim, Z = gd.generate_kalman_data(T, 0.8, 1, 0.4, 0.4)
X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)

model = FTheta()
loss_fn = torch.nn.MSELoss()
eta = 1e-3

for step in range(300):
    omd_step(model, loss_fn, X, k, eta)

    if step % 20 == 0:
        with torch.no_grad():
            y_pred = compute_predictions(model, X, k)
            loss = loss_fn(y_pred[k:], X[k:])
            print(step, loss.item())
