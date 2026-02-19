import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def simulate_msw_series(
    T=1000,
    a=1,         # small drift in log-price
    b1=0.15,         # influence of current x_t
    b2=0.1,         # influence of lagged x_{t-1}
    rho_y=0.6,      # autoregressive term for persistence in log-price
    sigma_y=0.3,   # smaller noise for realistic log-price scale
    alpha=0.5,
    beta=0.3,
    gamma=0.4,
    sigma_x=0.2,
    phi1=0.6,
    phi2=0.0,
    mu=(0.0, 1.0),
    sigma_c=(0.3, 0.8),
    P=np.array([[1.0, 0.0], [0.00, 1.0]]),
    init_z=0,
    seed=None,
    return_dataframe=False,
):
    if seed is not None:
        np.random.seed(seed)

    mu = np.asarray(mu)
    sigma_c = np.asarray(sigma_c)

    y = np.zeros(T)
    x = np.zeros(T)
    c = np.zeros(T)
    d = np.zeros(T)
    z = np.zeros(T, dtype=int)
    print(sigma_x)

    z[0] = int(init_z)
    c[0] = mu[z[0]] / (1 - phi1 - phi2) if abs(1 - phi1 - phi2) > 1e-6 else mu[z[0]]
    if T > 1:
        c[1] = c[0]

    x[0] = alpha / (1 - beta) if abs(1 - beta) > 1e-6 else 0.0
    y[0] = np.log(10.0)  # initial log-price around exp(2.3) ≈ 10

    def next_state(curr):
        return np.random.choice([0, 1], p=P[curr])

    for t in range(1, T):
        z[t] = next_state(z[t - 1])
        c_lag1 = c[t - 1]
        c_lag2 = c[t - 2] if t - 2 >= 0 else c_lag1
        eps_c = np.random.normal(scale=sigma_c[z[t]])
        c[t] = mu[z[t]] + phi1 * c_lag1 + phi2 * c_lag2 + eps_c

        eta_x = np.random.normal(scale=sigma_x)
        x[t] = alpha + beta * x[t - 1] + gamma * c[t] + eta_x

        e_y = np.random.normal(scale=sigma_y)
        x_lag1 = x[t - 1] if t - 1 >= 0 else x[t]
        y[t] = a + b1 * x[t] + b2 * x_lag1 + rho_y * y[t - 1] + e_y

    if return_dataframe:
        return pd.DataFrame({'y': y, 'x': x, 'c': c, 'z': z})
    r = np.diff(x)
    r = np.insert(r,0,0)
    print(sigma_x)
    return r, x, c, z


class SyntheticTradingEnv(gym.Env):
    """Gym environment for externally generated x_t, c_t series, with EWMA volatility estimate."""

    def __init__(self, x_series, c_series, alpha=0.05, kappa=5e-4, gamma=0.5):
        super().__init__()
        assert len(x_series) == len(c_series)
        self.x_series = x_series
        self.c_series = c_series
        self.T = len(x_series)
        self.alpha = alpha  # EWMA smoothing parameter
        self.kappa = kappa
        self.gamma = gamma

        # Continuous action space in [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation: [x_t, c_t]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 1  # start after first difference available
        self.a_prev = 0.0
        self.done = False
        self.sigma2_hat = 0.0
        obs = np.array([self.x_series[self.t], self.c_series[self.t]], dtype=np.float32)
        return obs, {}

    def _update_sigma2(self, x_prev, x_curr):
        r_t = x_curr - x_prev
        self.sigma2_hat = self.alpha * (r_t ** 2) + (1 - self.alpha) * self.sigma2_hat

    def step(self, action):
        a_t = float(np.clip(action[0], 0.0, 1.0))

        # Loop sequence when end is reached
        if self.t >= self.T - 2:
            self.t = 1
            self.a_prev = 0.0
            self.sigma2_hat = 0.0

        x_t, x_tp1 = self.x_series[self.t], self.x_series[self.t + 1]
        c_tp1 = self.c_series[self.t + 1]

        self._update_sigma2(x_t, x_tp1)

        #R = np.exp(x_tp1 - x_t) - 1.0  # realized return
        R = x_tp1 - x_t #use log returns
        reward = a_t * R - self.kappa * abs(a_t - self.a_prev) - self.gamma * (a_t ** 2) * self.sigma2_hat
        reward -= 0.05 * (a_t - self.a_prev)**2 #smoothness penalty
        #reward /= 0.1 #scale down typical reward variance

        # advance
        self.a_prev = a_t
        self.t += 1

        obs = np.array([x_tp1, c_tp1], dtype=np.float32)
        info = {"R": R, "sigma2": self.sigma2_hat}
        return obs, reward, False, False, info

# === Training example ===
def run_training_example(mu, sigma, seed=0):
    np.random.seed(seed)
    T = 1000

    # generate c_t
    c = np.zeros(T)
    e = np.random.normal(0, sigma, T)
    c[0:2] = 0.0
    for t in range(2, T):
        c[t] = mu + 0.6 * c[t - 1]  + e[t]

    # generate x_t
    x = np.zeros(T)
    eta = np.random.normal(0, 0.2, T)
    for t in range(1, T):
        x[t] = 0.5 + 0.3 * x[t - 1] + 0.4 * c[t] + eta[t]

    env = SyntheticTradingEnv(x, c)
    vec_env = DummyVecEnv([lambda: env])

    model = PPO("MlpPolicy", vec_env, verbose=0, policy_kwargs=dict(log_std_init=-1.0), learning_rate=1e-4, n_steps=1024, batch_size=256)
    model.learn(total_timesteps=300_000)

    # Evaluate policy behavior
    obs, _ = env.reset()
    actions = []
    for _ in range(500):  # twice through the data
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        actions.append(action[0])
        
    actions_smooth = pd.Series(actions).ewm(span=50).mean()
    plt.figure(figsize=(8, 3))
    plt.plot(actions_smooth, label=f"mu={mu}, sigma={sigma}")
    plt.title("Average learned allocation a_t")
    plt.xlabel("t")
    plt.ylabel("a_t")
    plt.legend()
    plt.tight_layout()
    plt.show()

#if __name__ == "__main__":
#    run_training_example(mu=0.0, sigma=0.3)
#    run_training_example(mu=1.0, sigma=0.8)
