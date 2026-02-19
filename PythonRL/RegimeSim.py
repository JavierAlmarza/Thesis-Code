import matplotlib.pyplot as plt
import modRegime as mR
import numpy as np
y, x, c, z = mR.simulate_msw_series(T=2000)

fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
axes[0].plot(z)
axes[0].set_ylabel('regime')
axes[1].plot(c)
axes[1].set_ylabel('climate variable (z)')
axes[2].plot(x)
axes[2].set_ylabel('latent variable')
axes[3].plot(y)
axes[3].set_ylabel('log price')
axes[4].plot(np.exp(x))
axes[4].set_ylabel('commodity price')
axes[4].set_xlabel('time')
plt.tight_layout()
plt.show()