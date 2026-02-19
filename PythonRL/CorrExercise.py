import numpy as np

# parameters
T = 2000

# generate independent white noises
e = np.random.normal(0, 1, T)
eta = np.random.normal(0, 1, T)

# generate random walks
x = np.cumsum(e)
y = np.cumsum(eta)

# sample correlation
rho = np.corrcoef(x, y)[0, 1]

print(f"Sample correlation ρ = {rho:.3f}")
