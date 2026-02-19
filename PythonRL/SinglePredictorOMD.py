import torch
import torch.nn as nn
import torch.optim as optim
import torch.func as stateless
import GenerateData as gd


# -------------------------
# Model definitions
# -------------------------

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


# -----------------------------------
# Optimal Mirror Descent Step
# -----------------------------------

def omd_step(model, loss_fn, X, k, eta):
    """
    Performs one Optimal Mirror Descent update:

        w_wait = w - eta * grad L(w)
        w_next = w - eta * grad L(w_wait)

    implemented via stateless.functional_call.
    """

    # 1) Compute grad L(w_t)
    y_pred = compute_predictions(model, X, k)
    loss = loss_fn(y_pred[k:], X[k:])
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # Move parameters into a flat dict
    param_dict = {name: p for name, p in model.named_parameters()}

    # 2) Construct w_wait = w - eta * grad
    w_wait = {name: param_dict[name] - eta * g
              for (name, g) in zip(param_dict.keys(), grads)}

    # 3) Compute grad L(w_wait)
    #y_pred_wait = stateless.functional_call(model, w_wait, (X, k), kwargs={})
    # But compute_predictions() expects a module, not a functional_call call,
    # so we wrap it:

    def model_wait_forward(inp):
        return stateless.functional_call(model, w_wait, (inp,))

    # Recompute predictions using the w_wait forward
    # — we need a version of compute_predictions that takes a forward fn
    y2 = torch.zeros_like(y_pred)
    y_prev = torch.tensor(0.0)
    T = X.shape[0]

    for t in range(k, T):
        inp = torch.tensor([X[t - k].item(), y_prev.item()], dtype=X.dtype)
        y_t = model_wait_forward(inp)
        y2[t] = y_t
        y_prev = y_t.detach()

    loss_wait = loss_fn(y2[k:], X[k:])
    grads_wait = torch.autograd.grad(loss_wait, w_wait.values())

    # 4) Final OMD update: w_next = w - eta * grad L(w_wait)
    with torch.no_grad():
        for (name, p), g in zip(model.named_parameters(), grads_wait):
            p -= eta * g


# -----------------------------
# Run training with OMD
# -----------------------------
T = 1000
k = 1

Xsim, Z = gd.generate_kalman_data(T, 0.8, 1, 0.4, 0.4)
X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)

model = FTheta()
loss_fn = nn.MSELoss()
eta = 1e-3

for step in range(150):   # 150 OMD steps behaves like 300 Adam steps
    omd_step(model, loss_fn, X, k, eta)

    if step % 20 == 0:
        with torch.no_grad():
            y_pred = compute_predictions(model, X, k)
            loss = loss_fn(y_pred[k:], X[k:])
            print(step, loss.item())
