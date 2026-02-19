import gymnasium as gym
import numpy as np
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch
import pandas as pd
import pyfolio as pf
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.distributions import make_proba_distribution
from scipy.optimize import minimize

class MarketReplayEnv(gym.Env):
    """
    Market replay environment with 11 sector returns and volatility signals.
    Reward: Differential Sharpe Ratio (Moody & Saffell 2001).
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, R, vol20, vol2060, vix, eta=0.05, window=60, episode_length=252*5):
        super(MarketReplayEnv, self).__init__()

        assert R.shape[0] == 11, "R must have shape (11, T)"
        T = R.shape[1]
        assert len(vol20) == T and len(vol2060) == T and len(vix) == T

        self.R = R
        self.vol20 = vol20
        self.vol2060 = vol2060
        self.vix = vix
        self.T = T
        self.window = window
        self.episode_length = episode_length
        self.eta = eta

        # Action space: R^11, unconstrained (softmax applied inside step)
        self.action_space = spaces.Box(low=-50.0, high=50.0, shape=(11,), dtype=np.float32)

        # Observation: 11*window past returns + 11 weights + 3 vol signals
        obs_dim = 11 * window + 11 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # State variables
        self.current_step = None
        self.start_idx = None
        # replace uniform init
        alpha = np.ones(11) * 1.5  # concentration parameter (tune this)
        self.weights_prev = self.np_random.dirichlet(alpha)
        
        self.A = 0.0
        self.B = 0.0
        self.logged_rewards = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # pick random start ensuring enough room for window + episode
        self.start_idx = self.np_random.integers(low=self.window, high=self.T - self.episode_length+1)
        self.current_step = self.start_idx
        # replace uniform init
        alpha = np.ones(11) * 1.5  # concentration parameter (tune this)
        self.weights_prev = self.np_random.dirichlet(alpha)
        print('Initial weights are ',self.weights_prev)


        # Initialize DSR stats (A, B) with small values
        
        
        #init_ret = float(np.dot(self.weights_prev, self.R[:, self.current_step]))
        #self.A = init_ret
        warm_start = max(0, self.start_idx - self.window)
        warm_slice = self.R[:, warm_start:self.start_idx]  # shape (n_assets, window)
        
        if warm_slice.shape[1] > 1:
            port_returns = (self.weights_prev.reshape(-1,1) * warm_slice).sum(axis=0)
            self.A = float(port_returns.mean())
            self.B = float(port_returns.var() + self.A**2)
            #print('Var is ',port_returns.var(),', B is ',self.B,', sample mean squared is ', self.A**2)
            #print('ep length is ',self.episode_length)
            #print('rewards length is ',len(self.logged_rewards))
            
        else:
            # fallback: at least not zero-variance
            #print('warm_slice is false, shape is ',warm_slice.shape[1])
            init_ret = float(np.dot(self.weights_prev, self.R[:, self.current_step]))
            self.A = init_ret
            self.B = init_ret**2 + 1e-6

        #self.B = init_ret ** 2
        
        
        self.logged_rewards = []        

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # map action to weights via softmax
        exp_a = np.exp(action - np.max(action))
        weights = exp_a / exp_a.sum()

        # portfolio return at t
        ret = float(np.dot(weights, self.R[:, self.current_step]))
        

        # update A, B (exponential moving averages)
        A_prev, B_prev = self.A, self.B
        self.A = (1 - self.eta) * self.A + self.eta * ret
        self.B = (1 - self.eta) * self.B + self.eta * (ret ** 2)

        # compute ΔA, ΔB (normalized by eta, as in Moody & Saffell eq. 14)
        deltaA = (self.A - A_prev) / self.eta
        deltaB = (self.B - B_prev) / self.eta

        denom = B_prev - A_prev**2
        if denom<1e-8:
            print('DENOM IS TINYYYYY')
        denom = max(denom, 1e-8)  # clip to avoid zero/negative variance
        dsr = (B_prev * deltaA - 0.5 * A_prev * deltaB) / (denom ** 1.5)

        reward = dsr
        self.logged_rewards.append(reward)
        avgrew = sum(self.logged_rewards)/len(self.logged_rewards)
        
        if abs(reward)>10:
            print('A is ',A_prev,', B is ',B_prev,', ret is ',ret,', rew is ',reward)
            print('Length is ',len(self.logged_rewards) )
            print('Step is ', self.current_step,', Avg reward is ',avgrew)

        self.weights_prev = weights
        info = {"portfolio_return": ret, "weights": weights, "DSR": dsr}
        
        obs = self._get_observation()
        self.current_step += 1
        done = self.current_step >= self.start_idx + self.episode_length
        if done:
            self.logged_rewards = []
            print('Done. Weights are ',weights)



        return obs, reward, done, False, info

    def _get_observation(self):
        # last `window` returns for all assets, flattened
        past_returns = self.R[:, self.current_step - self.window:self.current_step].flatten()

        # last action
        last_action = self.weights_prev

        # volatility features
        vols = np.array([
            float(self.vol20[self.current_step]),
            float(self.vol2060[self.current_step]),
            float(self.vix[self.current_step]),
        ])

        return np.concatenate([past_returns, last_action, vols]).astype(np.float32)

    def render(self, mode="human"):
        print(f"Step {self.current_step}, Last portfolio return: {self.A:.6f}, Sharpe approx")
        
# put this in a file or run in your notebook after importing modFin2


class DeterministicValidationEnv(MarketReplayEnv):
    """
    MarketReplayEnv where reset() always starts at the SAME index:
    start_idx := window, so initial obs uses the preceding `window` days
    and the episode then runs for `episode_length` steps deterministically.
    Use this class by passing R sliced as R[:, T:T+window+episode_length],
    and same for vol arrays.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # enforce deterministic behavior for reset
        # (we'll set start_idx to exactly 'window' inside reset)

    def reset(self, seed=None, options=None):
        # Gymnasium reset signature
        super().reset(seed=seed)

        # Deterministic start: always begin at self.window
        self.start_idx = self.window
        self.current_step = self.start_idx

        # replace uniform init
        alpha = np.ones(11) * 1.5  # concentration parameter (tune this)
        self.weights_prev = self.np_random.dirichlet(alpha)

        # --- Warm-up stats for A and B (same logic as stochastic env) ---
        warm_start = max(0, self.start_idx - self.window)
        warm_slice = self.R[:, warm_start:self.start_idx]  # shape (n_assets, window)

        if warm_slice.shape[1] > 1:
            port_returns = (self.weights_prev.reshape(-1, 1) * warm_slice).sum(axis=0)
            self.A = float(port_returns.mean())
            self.B = float((port_returns**2).mean())
        else:
            # fallback in case window=1 or no slice available
            init_ret = float(np.dot(self.weights_prev, self.R[:, self.current_step]))
            self.A = init_ret
            self.B = init_ret**2 + 1e-6

        # Return initial observation
        return self._get_observation(), {}
        

def evaluate_agents_sharpe(ppo_agents, R_full, vol20a, vol2060, vix1,
                           T, window=60, val_episode_length=252, verbose=True):
    """
    Evaluate each model in ppo_agents on the same deterministic validation episode
    that starts at index T (i.e., uses data R_full[:, T:T+window+val_episode_length]).

    Returns:
      best_idx, results_list
      where results_list is list of dicts with keys:
         'idx', 'sharpe', 'daily_returns' (np.array shape (val_episode_length,)), 'weights' (list per day)
    """
    results = []
    # prepare the validation data slice
    R_val = R_full[:, T : T + window + val_episode_length]
    vol20_val = vol20a[T : T + window + val_episode_length]
    vol2060_val = vol2060[T : T + window + val_episode_length]
    vix_val = vix1[T : T + window + val_episode_length]

    for i, model in enumerate(ppo_agents):
        # create a fresh env for each agent so internal stats A,B are independent
        env = DeterministicValidationEnv(R_val, vol20_val, vol2060_val, vix_val,
                                         eta=0.01, window=window, episode_length=val_episode_length)
        obs, _ = env.reset()

        done = False
        truncated = False
        daily_returns = []
        weights_history = []

        # run deterministic policy
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            # collect portfolio return (info provided by your env)
            # if your env uses a different key, change this line accordingly
            rt = info.get("portfolio_return", None)
            w = info.get("weights", None)
            if rt is None:
                # if portfolio return not in info, try to compute from weights and R
                # but prefer having info["portfolio_return"] available
                raise KeyError("env.info must contain 'portfolio_return'")
            daily_returns.append(rt)
            weights_history.append(w)

        daily_returns = np.array(daily_returns, dtype=float)
        # compute annualized Sharpe (classical): mean / std * sqrt(252)
        if daily_returns.size == 0:
            sharpe = -np.inf
        else:
            std = np.std(daily_returns, ddof=1)
            if std == 0:
                sharpe = -np.inf
            else:
                sharpe = np.mean(daily_returns) / std * np.sqrt(252.0)

        if verbose:
            print(f"Agent {i}: Sharpe (ann.) = {sharpe:.4f}, length={len(daily_returns)}")

        results.append({
            "idx": i,
            "sharpe": sharpe,
            "daily_returns": daily_returns,
            "weights": weights_history,
        })

    # pick best by sharpe
    best_idx = max(results, key=lambda r: r["sharpe"])["idx"]
    if verbose:
        print(f"Best agent index: {best_idx} with Sharpe = {results[best_idx]['sharpe']:.4f}")

    return best_idx, results


