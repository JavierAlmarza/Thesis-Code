import gymnasium as gym
import numpy as np
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    def __init__(self, R, q=5, r=0.01, k=0.05):
        """
        R: np.array of shape (N, T), time series batches
        q: number of return lags (int)
        r: safe asset return (float)
        k: penalty on squared return (float)
        """
        super().__init__()
        self.R = R
        self.N, self.T = R.shape
        self.q = q
        self.r = r
        self.k = k

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0] + [-np.inf]*q, dtype=np.float32),
            high=np.array([1.0] + [np.inf]*q, dtype=np.float32),
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.batch_idx = self.np_random.integers(self.N)  # pick random path
        self.t = self.q  # start after we can form full lag state
        self.prev_w = 0.5  # default starting weight
        self.done = False
        #self.terminated = False


        # Build initial state: (w_prev, R_{t-1}, ..., R_{t-q})
        R_slice = self.R[self.batch_idx, self.t - self.q:self.t][::-1]
        self.state = np.concatenate(([self.prev_w], R_slice)).astype(np.float32)
        return self.state, {}

    def step(self, action):
        w = float(np.clip(action[0], 0.0, 1.0))
        R_t = self.R[self.batch_idx, self.t]

        reward = w * R_t + self.r * (1 - w) - self.k * R_t**2

        # Update state
        R_lags = self.state[1:self.q+1]  # previous lags
        new_state = np.concatenate(([w], [R_t], R_lags[:-1]))
        self.state = new_state.astype(np.float32)

        self.prev_w = w
        self.t += 1
        self.done = (self.t >= self.T)
        #self.terminated = (self.t >= self.T)

        return self.state, reward, self.done, False, {}

    def render(self):
        print(f"t={self.t}, state={self.state}")
