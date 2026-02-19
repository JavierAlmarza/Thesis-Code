import torch
import torch.nn as nn
import torch.optim as optim
import GenerateData as gd


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


class FTheta2(nn.Module):
    """
    Neural net implementing F_theta: R^2 -> R with architecture 2 -> 8 -> 8 -> 1
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x_in):
        return self.net(x_in)


def compute_predictions(model, X, k):
    T = X.shape[0]
    y = torch.zeros(T)
    y_prev = torch.tensor(0.0)

    for t in range(k, T):
        # safe (2,) tensor
        inp = torch.tensor([X[t - k].item(), y_prev.item()], dtype=X.dtype)
        y_t = model(inp)
        y[t] = y_t
        y_prev = y_t.detach()

    return y



# -----------------------------
# Example usage / training stub
# -----------------------------
T = 1000
k = 0   
#X = torch.randn(T)

Xsim, Z = gd.generate_kalman_data(T,0.8,1,0.4,0.4)
X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32, device='cpu')

model = FTheta()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for step in range(300):
    optimizer.zero_grad()

    y_pred = compute_predictions(model, X, k)

    # Use MSE between predicted y_t and the true x_t
    # but only for t >= k since others are undefined
    loss = ((y_pred[k:] - X[k:]) ** 2).mean()

    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(step, loss.item())
