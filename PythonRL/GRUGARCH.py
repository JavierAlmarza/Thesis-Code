import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Generate synthetic GARCH(1,1)
# -------------------------------
def generate_garch(T=2000, omega=0.1, alpha=0.1, beta=0.8):
    x = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, T):
        x[t] = np.sqrt(sigma2[t-1]) * np.random.randn()
        sigma2[t] = omega + alpha * x[t]**2 + beta * sigma2[t-1]
    return x, np.sqrt(sigma2)

x, sigma = generate_garch()

# normalize
x = (x - x.mean()) / x.std()

# -------------------------------
# 2. Define RVAE-style model
# -------------------------------
class LatentVolatilityModel(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.f = nn.GRUCell(1, hidden_dim)
        self.to_h = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        T = len(x)
        h_rnn = torch.zeros(1, self.f.hidden_size)
        h_logvar = []
        for t in range(1, T):
            h_rnn = self.f(x[t-1].view(1,1), h_rnn)
            h = self.to_h(h_rnn)
            h_logvar.append(h)
        return torch.cat(h_logvar, dim=0).squeeze(-1)

model = LatentVolatilityModel(hidden_dim=16)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------
# 3. Train on log-likelihood
# -------------------------------
x_torch = torch.tensor(x, dtype=torch.float32)

for epoch in range(200):
    optimizer.zero_grad()
    h_logvar = model(x_torch)
    # p(x_t|h_t) = N(0, exp(h_t))
    ll = -0.5 * (h_logvar + (x_torch[1:]**2) / torch.exp(h_logvar))
    loss = -ll.mean()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | NLL {loss.item():.4f}")

# -------------------------------
# 4. Inspect learned volatility
# -------------------------------
with torch.no_grad():
    h_logvar = model(x_torch)
    sigma_hat = torch.exp(0.5 * h_logvar).numpy()

plt.figure(figsize=(10,4))
plt.plot(sigma, label='True σ_t', alpha=0.7)
plt.plot(sigma_hat, label='Learned σ̂_t', alpha=0.7)
plt.legend()
plt.title("Recovery of conditional volatility")
plt.tight_layout()
plt.show()
