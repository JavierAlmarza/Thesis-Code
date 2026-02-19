import gymnasium as gym
import numpy as np
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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


def generate_cccmultivarch(N, T, seed=None):
    """
    Generate N paths of length T of a bivariate CCC-ARCH(1) model using

    sigma_{i,t}^2 = omega_i + alpha_{i,1} eps_{1,t-1}^2 + alpha_{i,2} eps_{2,t-1}^2
    Sigma_t = [[sigma1^2, rho*sigma1*sigma2],
               [rho*sigma1*sigma2, sigma2^2]]
    eps_t = Sigma_t^{1/2} @ z_t, z_t ~ iid N(0,1)

    Returns:
        paths: np.array of shape (N, T, 2)
    """
    if seed is not None:
        np.random.seed(seed)


    # Parameters

    mu1 = 0.15 / 252
    mu2 = 0.06 / 252
    omega = np.array([0.0004, 0.00008])
    A = np.array([[0.15, 0.06],
                  [0.04, 0.1]])
    rho = 0.5

    paths = np.zeros((N, T, 2))

    def matrix_sqrt_2x2(a, b, c):
        """Principal square root of 2x2 symmetric matrix [[a,c],[c,b]]"""
        delta = np.sqrt(a*b - c**2)
        s = np.sqrt(a + b + 2*delta)
        sqrt_matrix = np.array([[a + delta, c],
                                [c, b + delta]]) / s
        return sqrt_matrix

    for n in range(N):
        sigma2 = np.zeros((T,2))
        eps = np.zeros((T,2))
        sigma2[0] = omega.copy()
        z = np.random.normal(size=(T,2))

        for t in range(1,T):
            eps_prev = eps[t-1]
            sigma2[t,0] = omega[0] + A[0,0]*eps_prev[0]**2 + A[0,1]*eps_prev[1]**2
            sigma2[t,1] = omega[1] + A[1,0]*eps_prev[0]**2 + A[1,1]*eps_prev[1]**2

            s1 = np.sqrt(sigma2[t,0])
            s2 = np.sqrt(sigma2[t,1])
            Sigma_t_sqrt = matrix_sqrt_2x2(s1**2, s2**2, rho*s1*s2)

            eps[t] = Sigma_t_sqrt @ z[t]

        paths[n,:,0] = mu1 + eps[:,0]
        paths[n,:,1] = mu2 + eps[:,1]

    return paths

class PortfolioEnv2(gym.Env):
    def __init__(self, R, q=5, k=0.1, include_action=True, include_vol=True):
        """
        Portfolio allocation environment with two risky assets.
        
        Args:
            R : ndarray of shape (N, T, 2) 
                Simulated returns (N paths, T periods, 2 assets)
            q : int
                Number of past lags to include in the state
            k : float
                Quadratic penalty coefficient
            include_action : bool
                Whether to include last action in the state
            include_vol : bool
                Whether to include rolling portfolio volatility in the state
        """
        super().__init__()
        self.R = R
        self.N, self.T, self.d = R.shape
        assert self.d == 2, "R must have 2 assets"
        self.q = q
        self.k = k
        self.include_action = include_action
        self.include_vol = include_vol
        self.logged_actions = []
        self.last_action = None

        # --- Construct observation vector components
        obs_low = []
        obs_high = []

        if include_action:
            obs_low.append(0.0)
            obs_high.append(1.0)

        if q > 0:
            obs_low += [-np.inf] * (2 * q)  # 2 assets per lag
            obs_high += [np.inf] * (2 * q)

        if include_vol:
            obs_low.append(0.0)
            obs_high.append(np.inf)

        if len(obs_low) == 0:
            self.use_dummy_obs = True
            obs_low = [0.0]
            obs_high = [0.0]
        else:
            self.use_dummy_obs = False

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.batch_idx = self.np_random.integers(self.N)
        self.t = self.q
        self.prev_w = 0.5
        self.done = False
        self.terminated = False
        self.ret_buffer = []

        # Lags: shape (q,2), flatten most recent first
        if self.q > 0:
            R_slice = self.R[self.batch_idx, self.t - self.q:self.t][::-1].flatten()
        else:
            R_slice = []

        vol = 0.0
        a = [self.prev_w] if self.include_action else []
        b = [vol] if self.include_vol else []
        state_parts = a + list(R_slice) + b

        if self.use_dummy_obs:
            self.state = np.zeros(1, dtype=np.float32)
        else:
            self.state = np.array(state_parts, dtype=np.float32)

        return self.state, {}

    def step(self, action):
        w = float(np.clip(action, 0.0, 1.0))
        R_t = self.R[self.batch_idx, self.t]  # shape (2,)
        self.logged_actions.append(w)
        self.last_action = w

        # portfolio return
        port_ret = w * R_t[0] + (1 - w) * R_t[1]

        # reward with quadratic penalty
        reward = port_ret - self.k * port_ret**2

        # update rolling volatility buffer
        self.ret_buffer.append(port_ret)
        if len(self.ret_buffer) > 30:
            self.ret_buffer.pop(0)
        vol = np.std(self.ret_buffer) if len(self.ret_buffer) >= 2 else 0.0

        # lagged returns (shift)
        if self.q > 0:
            # shift previous state returns, prepend current R_t
            lag_start = 1 if self.include_action else 0
            old_lags = self.state[lag_start:lag_start + 2*(self.q-1)]
            lagged_returns = np.concatenate((R_t, old_lags))
        else:
            lagged_returns = []

        a = [w] if self.include_action else []
        b = [vol] if self.include_vol else []
        state_parts = a + list(lagged_returns) + b

        if self.use_dummy_obs:
            self.state = np.zeros(1, dtype=np.float32)
        else:
            self.state = np.array(state_parts, dtype=np.float32)

        self.prev_w = w
        self.t += 1
        self.done = self.terminated = (self.t >= self.T)

        return self.state, reward, self.done, False, {}

    def render(self):
        print(f"t={self.t}, state={self.state}")

class SimpleARCH:
    def __init__(self, p=1):
        self.p = p
        
    def fit(self, data):
        self.data = np.asarray(data)
        def neg_loglik(params):
            omega, alpha = params
            T = len(self.data)
            sigma2 = np.zeros(T)
            sigma2[0] = np.var(self.data)
            for t in range(1, T):
                sigma2[t] = omega + alpha * self.data[t-1]**2
            ll = 0.5 * (np.log(2*np.pi) + np.log(sigma2) + self.data**2 / sigma2)
            return ll.sum()
        
        bounds = [(1e-6, None), (0, None)]
        res = minimize(neg_loglik, x0=[0.1, 0.1], bounds=bounds)
        self.params = res.x
        return self

class SigmoidGaussianDistribution(Distribution):

    def __init__(self, mu: torch.Tensor, log_std: torch.Tensor):
        super().__init__()
        self.action_dim =  mu.shape[-1]  
        self.mu = mu
        log_stdv = torch.clamp(log_std, min=-300, max=200)
        self.std = torch.exp(log_stdv)
        self.epsilon = 1e-6
        self.margin = 0.05
        if torch.isnan(self.mu).any() or torch.isnan(self.std).any():
            raise ValueError(f"NaN in distribution params:\nmu={self.mu}\nstd={self.std}")
        self.distribution = Normal(self.mu, self.std)   
        
    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        # This creates the output layer for the policy
        return nn.Linear(latent_dim, 2 * self.action_dim)

    @classmethod
    def proba_distribution(cls, mean_actions: torch.Tensor, log_std: torch.Tensor):
        return cls(mean_actions, log_std)

#    @classmethod
#    def proba_distribution(cls, mean_actions: torch.Tensor, log_std: torch.Tensor):
      
#        obj = cls.__new__(cls)  # create instance without __init__
#        obj.mu = mean_actions
#        obj.std = torch.exp(log_std)
#        obj.epsilon = 1e-6
#        return obj

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # Invert sigmoid: u = log(a / (1 - a))
        #u = torch.log((actions-(-self.margin)) / ((1+self.margin) - actions + self.epsilon) + self.epsilon)
        #log_prob_u = Normal(self.mu, self.std).log_prob(u)
        squashed = (actions - self.margin) / (1 - 2 * self.margin)
        squashed = torch.clamp(squashed, self.epsilon, 1 - self.epsilon)
        u = torch.log(squashed / (1 - squashed))

        log_prob_u = self.distribution.log_prob(u)
        log_det_jacobian = torch.log((1+2*self.margin)*actions * (1 - actions) + self.epsilon)
        return log_prob_u - log_det_jacobian

    def entropy(self) -> torch.Tensor:
        # Entropy of the base Gaussian — underestimates entropy after squashing
        return Normal(self.mu, self.std).entropy()
        #return (0.5 + 0.5 * torch.log(torch.tensor(2 * torch.pi)+self.epsilon) + torch.log(self.std+self.epsilon)).sum(dim=1)
      

    def sample(self) -> torch.Tensor:
        u = Normal(self.mu, self.std).rsample()
        return -self.margin+(1+2*self.margin)*torch.sigmoid(u)

    def mode(self) -> torch.Tensor:
        return torch.sigmoid(self.mu)
        
    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()

    def get_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.log_prob(actions)
        
    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self.proba_distribution(mean_actions, log_std)
        return self.get_log_prob(actions)

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, squash_output=False)

    def make_dist(self, action_dim: int):
        return SigmoidGaussianDistribution

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Replace distribution
        self.action_dist = self.make_dist(self.action_space.shape[0])
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = nn.Linear(latent_dim_pi, 2 * self.action_space.shape[0])

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        params = self.action_net(latent_pi)
        # SigmoidGaussianDistribution.proba_distribution expects (mean, log_std)
        action_dim = self.action_space.shape[0]
        mean_actions = params[:, :action_dim]
        log_std = params[:, action_dim:]
        return self.action_dist.proba_distribution(mean_actions, log_std)



class SquashedGaussianPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         net_arch=dict(pi=[64, 64], vf=[64, 64]),
                         activation_fn=nn.Tanh)

    def make_dist(self, action_dim):
        return SquashedDiagGaussianDistribution(action_dim)



def generate_gaussian(N, T, mu=0.0, sigma=1.0):
    return np.random.normal(loc=mu, scale=sigma, size=(N, T))


def generate_arch1(N, T, alpha0, alpha1):

    #Generate N simulations of an ARCH(1) process of length T.

    r = np.zeros((N, T))
    sigma2 = np.zeros((N, T))

    # Initial conditional variance
    sigma2[:, 0] = alpha0 / (1 - alpha1) #stationary value

    # Generate standard normal shocks
    z = np.random.randn(N, T)

    # First return
    r[:, 0] = np.sqrt(sigma2[:, 0]) * z[:, 0]

    for t in range(1, T):
        sigma2[:, t] = alpha0 + alpha1 * r[:, t-1]**2
        r[:, t] = np.sqrt(sigma2[:, t]) * z[:, t]

    return r

def linear_lr_schedule(progress_remaining):
    return 1e-5 + (3e-4 - 1e-5) * progress_remaining

class PortfolioEnv(gym.Env):
    def __init__(self, R, q=5, r=0.01, k=0.05, include_action=True, include_vol=True):
        super().__init__()
        self.R = R
        self.N, self.T = R.shape
        self.q = q
        self.r = r
        self.k = k
        self.include_action = include_action
        self.include_vol = include_vol
        self.logged_actions = []
        self.last_action = None

        # --- Construct observation vector components
        obs_low = []
        obs_high = []

        if include_action:
            obs_low.append(0.0)
            obs_high.append(1.0)
        if q > 0:
            obs_low += [-np.inf] * q
            obs_high += [np.inf] * q
        if include_vol:
            obs_low.append(0.0)
            obs_high.append(np.inf)

        # Handle empty observation vector: add dummy dimension
        if len(obs_low) == 0:
            self.use_dummy_obs = True
            obs_low = [0.0]
            obs_high = [0.0]
        else:
            self.use_dummy_obs = False

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.batch_idx = self.np_random.integers(self.N)
        self.t = self.q
        self.prev_w = 0.5
        self.done = False
        self.terminated = False
        self.ret_buffer = []

        R_slice = self.R[self.batch_idx, self.t - self.q:self.t][::-1] if self.q > 0 else []
        vol = 0.0

        a = [self.prev_w] if self.include_action else []
        b = [vol] if self.include_vol else []
        state_parts = a + list(R_slice) + b

        if self.use_dummy_obs:
            self.state = np.zeros(1, dtype=np.float32)
        else:
            self.state = np.array(state_parts, dtype=np.float32)

        return self.state, {}

    def step(self, action):
        print('Passed action is ', action)
        w = float(np.clip(action, 0.0, 1.0))
        R_t = self.R[self.batch_idx, self.t]
        self.logged_actions.append(w)
        self.last_action = w

        reward = w * R_t + self.r * (1 - w) - self.k * (w * R_t) ** 2
        print('Reward is ', reward)

        self.ret_buffer.append(R_t)
        if len(self.ret_buffer) > 30:
            self.ret_buffer.pop(0)
        vol = np.std(self.ret_buffer) if len(self.ret_buffer) >= 2 else 0.0

        if self.q > 0:
            # Extract past lags
            lag_start = 1 if self.include_action else 0
            R_lags = np.concatenate(([R_t], self.state[lag_start:lag_start + self.q - 1]))  # shift lags
            lagged_returns = R_lags
        else:
            lagged_returns = []

        a = [w] if self.include_action else []
        b = [vol] if self.include_vol else []
        state_parts = a + list(lagged_returns) + b

        if self.use_dummy_obs:
            self.state = np.zeros(1, dtype=np.float32)
        else:
            self.state = np.array(state_parts, dtype=np.float32)

        self.prev_w = w
        self.t += 1
        self.done = self.terminated = (self.t >= self.T)

        return self.state, reward, self.done, False, {}

    def render(self):
        print(f"t={self.t}, state={self.state}")


class SMPortfolioEnv(gym.Env):
    def __init__(self, R, q=5, r=0.01, k=0.05, include_action=True, include_vol=True):
        super().__init__()
        self.R = R
        self.N, self.T = R.shape
        self.q = q
        self.r = r
        self.k = k
        self.include_action = include_action
        self.include_vol = include_vol
        self.logged_actions = []
        self.last_action = None

        # --- Construct observation vector components
        obs_low = []
        obs_high = []

        if include_action:
            obs_low.append(0.0)
            obs_high.append(1.0)
        if q > 0:
            obs_low += [-np.inf] * q
            obs_high += [np.inf] * q
        if include_vol:
            obs_low.append(0.0)
            obs_high.append(np.inf)

        # Handle empty observation vector: add dummy dimension
        if len(obs_low) == 0:
            self.use_dummy_obs = True
            obs_low = [0.0]
            obs_high = [0.0]
        else:
            self.use_dummy_obs = False

        self.action_space = spaces.Box(low=np.array([-10000] * 1, dtype=np.float32), high= np.array([10000] * 1, dtype=np.float32), shape=(1,))
        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.batch_idx = self.np_random.integers(self.N)
        self.t = self.q
        self.prev_w = 0.5
        self.done = False
        self.terminated = False
        self.ret_buffer = []

        R_slice = self.R[self.batch_idx, self.t - self.q:self.t][::-1] if self.q > 0 else []
        vol = 0.0

        a = [self.prev_w] if self.include_action else []
        b = [vol] if self.include_vol else []
        state_parts = a + list(R_slice) + b

        if self.use_dummy_obs:
            self.state = np.zeros(1, dtype=np.float32)
        else:
            self.state = np.array(state_parts, dtype=np.float32)

        return self.state, {}

    def step(self, action):
        print('Passed action is ', action)
        #w = float(np.clip(action, 0.0, 1.0))
        R_t = self.R[self.batch_idx, self.t]
        exp_act = np.exp(action)
        smw = exp_act/(1+exp_act.sum())
        print('passed weight is ',smw)
        self.logged_actions.append(smw[0])
        self.last_action = smw[0]
        
        #logits = torch.tensor(action)  # action is shape (2,), e.g., from your policy network
        #probabilities = F.softmax(logits, dim=0)


        reward = smw[0] * R_t + self.r * (1-smw[0]) - self.k * (smw[0] * R_t) ** 2
        print('Reward is ', reward)

        self.ret_buffer.append(R_t)
        if len(self.ret_buffer) > 30:
            self.ret_buffer.pop(0)
        vol = np.std(self.ret_buffer) if len(self.ret_buffer) >= 2 else 0.0

        if self.q > 0:
            # Extract past lags
            lag_start = 1 if self.include_action else 0
            R_lags = np.concatenate(([R_t], self.state[lag_start:lag_start + self.q - 1]))  # shift lags
            lagged_returns = R_lags
        else:
            lagged_returns = []

        a = [smw[0]] if self.include_action else []
        b = [vol] if self.include_vol else []
        state_parts = a + list(lagged_returns) + b

        if self.use_dummy_obs:
            self.state = np.zeros(1, dtype=np.float32)
        else:
            self.state = np.array(state_parts, dtype=np.float32)

        self.prev_w = smw[0]
        self.t += 1
        self.done = self.terminated = (self.t >= self.T)

        return self.state, reward, self.done, False, {}

    def render(self):
        print(f"t={self.t}, state={self.state}")

class FinModelCallback(BaseCallback):
    def __init__(self, plot_freq=1000, saveName = "training_plot.png", verbose=0):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.saveName = saveName
        self.rewards = []
        self.actions = []
        self.means = []
        self.sds = []
        self.timesteps = []

    def _on_step(self) -> bool:
        # Record reward
        if "rewards" in self.locals:
            reward = self.locals["rewards"]
            # reward can be vectorized (n_envs,)
            if isinstance(reward, (list, np.ndarray)):
                reward = reward[0]  # keep first env for printing
            self.rewards.append(reward)

        if "actions" in self.locals:
            obs = self.locals["new_obs"]  # shape: (n_envs, obs_dim) in VecEnv
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.model.device)

            # Ensure 2D: [batch_size, obs_dim]
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            print("obs_tensor shape:", obs_tensor.shape)

            with torch.no_grad():
                features = self.model.policy.extract_features(obs_tensor)
                dist = self.model.policy.get_distribution(features)
                
                mu = dist.distribution.mean.cpu().numpy()
                sigma = dist.distribution.stddev.cpu().numpy()
                
            print("Mu:", mu, "Mu0:", mu[0], "mudim is", np.ndim(mu))
            print("Sigma:", sigma)

            self.means.append(mu[0][0] if np.ndim(mu) > 1 else mu[0])
            self.sds.append(sigma[0] if np.ndim(sigma) > 0 else sigma)

            action = self.locals["actions"]
            if isinstance(action, (list, np.ndarray)):
                action_to_log = action[0]
            else:
                action_to_log = action
            print("raw action is", action_to_log)
            self.actions.append(action_to_log)

        self.timesteps.append(self.num_timesteps)

        # Optional: plot periodically (commented out)
        # if self.n_calls % self.plot_freq == 0:
        #     self._plot_metrics()

        return True

    def _on_training_end(self) -> None:
        # Save plots once at the end
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

        ax1.plot(self.timesteps, self.rewards, label='Reward')
        ax1.set_ylabel('Reward')
        ax1.set_title('Step Rewards')
        ax1.legend()

        ax2.plot(self.timesteps, self.means, label='Means', color='green')
        ax2.plot(self.timesteps, self.sds, label='SDs', color='red')  # optional
        ax2.set_ylabel('Action')
        ax2.set_xlabel('Timestep')
        ax2.set_title('Sampled Actions')
        ax2.legend()

        fig.tight_layout()
        fig.savefig(self.saveName)  # saves to current working directory
        plt.close(fig) 

class FinModelCallback2(BaseCallback):
    def __init__(self, plot_freq=1000, saveName = "training_plot.png", verbose=0):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.saveName = saveName
        self.rewards = []
        self.actions = []
        self.means = []
        self.sds = []
        self.timesteps = []

    def _on_step(self) -> bool:
        # Record reward
        if "rewards" in self.locals:
            reward = self.locals["rewards"]
            # reward can be vectorized (n_envs,)
            if isinstance(reward, (list, np.ndarray)):
                reward = reward[0]  # keep first env for printing
            self.rewards.append(reward)

        if "actions" in self.locals:
            obs = self.locals["new_obs"]  # shape: (n_envs, obs_dim) in VecEnv
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.model.device)

            # Ensure 2D: [batch_size, obs_dim]
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            print("obs_tensor shape:", obs_tensor.shape)

            with torch.no_grad():
                features = self.model.policy.extract_features(obs_tensor)
                dist = self.model.policy.get_distribution(features)
                
                mu = dist.distribution.mean.cpu().numpy()
                sigma = dist.distribution.stddev.cpu().numpy()
                
            print("Mu:", mu, "Mu0:", mu[0], "mudim is", np.ndim(mu))
            print("Sigma:", sigma)

            self.means.append(mu[0][0] if np.ndim(mu) > 1 else mu[0])
            self.sds.append(sigma[0] if np.ndim(sigma) > 0 else sigma)

            action = self.locals["actions"]
            if isinstance(action, (list, np.ndarray)):
                action_to_log = action[0]
            else:
                action_to_log = action
            print("raw action is", action_to_log)
            self.actions.append(action_to_log)

        self.timesteps.append(self.num_timesteps)

        # Optional: plot periodically (commented out)
        # if self.n_calls % self.plot_freq == 0:
        #     self._plot_metrics()

        return True

    def _on_training_end(self) -> None:
        # Save plots once at the end
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

        ax1.plot(self.timesteps, self.rewards, label='Reward')
        ax1.set_ylabel('Reward')
        ax1.set_title('Step Rewards')
        ax1.legend()

        ax2.plot(self.timesteps, self.means, label='Means', color='green')
        ax2.plot(self.timesteps, self.sds, label='SDs', color='red')  # optional
        ax2.set_ylabel('Action')
        ax2.set_xlabel('Timestep')
        ax2.set_title('Sampled Actions')
        ax2.legend()

        fig.tight_layout()
        fig.savefig(self.saveName)  # saves to current working directory
        plt.close(fig)


class FinModelCallback0(BaseCallback):
    def __init__(self, plot_freq=1000, verbose=0):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.rewards = []
        self.actions = []
        self.means = []
        self.sds = []
        self.timesteps = []

    #def _init_callback(self) -> None:
        # Initialize a persistent figure and axes for real-time plotting
        #plt.ion()
        #self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    def _on_step(self) -> bool:
        # Record reward and action
        if "rewards" in self.locals:
            reward = self.locals["rewards"][0]
            self.rewards.append(reward)

        if "actions" in self.locals:
            obs=self.locals["new_obs"]
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.model.device)
            with torch.no_grad():
                print("obs_tensor shape:", obs_tensor.shape)
                obs_tensor = obs_tensor.squeeze(-1) 
                print("obs_tensor shape:", obs_tensor.shape)
                dist = self.model.policy.get_distribution(obs_tensor)
                mu = dist.distribution.mean.cpu().numpy()
                sigma = dist.distribution.stddev.cpu().numpy()  # log_std.exp() usually
                # Log them
            print("Mu:", mu, "Mu0:", mu[0],"mudim is ", np.ndim(mu))
            print("Sigma:", sigma)
            self.means.append(mu[0][0] if np.ndim(mu) > 0 else mu)
            self.sds.append(sigma[0] if np.ndim(sigma) > 0 else sigma)
            
            action = self.locals["actions"]
            action_space = self.model.action_space  # Should be Box(0.0, 1.0)
            print('raw action is ', action)
            self.actions.append(action[0] if np.ndim(action) > 0 else action)

        self.timesteps.append(self.num_timesteps)

        # Plot periodically
        #if self.n_calls % self.plot_freq == 0:
        #    self._plot_metrics()

        return True

    def _on_training_end(self) -> None:
        # Save plots once at the end
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

        ax1.plot(self.timesteps, self.rewards, label='Reward')
        ax1.set_ylabel('Reward')
        ax1.set_title('Step Rewards')
        ax1.legend()

        ax2.plot(self.timesteps, self.means, label='Means', color='green')
        # ax2.plot(self.timesteps, self.sds, label='SDs', color='red')  # optional
        ax2.set_ylabel('Action')
        ax2.set_xlabel('Timestep')
        ax2.set_title('Sampled Actions')
        ax2.legend()

        fig.tight_layout()
        fig.savefig("training_plot.png")  # saves to current working directory
        plt.close(fig) 

#    def _plot_metrics(self):
#        self.ax1.clear()
#        self.ax2.clear()

#        self.ax1.plot(self.timesteps, self.rewards, label='Reward')
#        self.ax1.set_ylabel('Reward')
#        self.ax1.set_title('Step Rewards')
#        self.ax1.legend()

#        self.ax2.plot(self.timesteps, self.means, label='Means', color='green')
        #self.ax2.plot(self.timesteps, self.sds, label='SDs', color='red')
#        self.ax2.set_ylabel('Action')
#        self.ax2.set_title('Sampled Actions')
#        self.ax2.set_xlabel('Timestep')
#        self.ax2.legend()

#        self.fig.tight_layout()
#        self.fig.canvas.draw()
#        self.fig.canvas.flush_events()
        
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class FinModelWithClipPenaltyCallback(FinModelCallback):
    def __init__(self, plot_freq=1000, verbose=0):
        super().__init__(plot_freq=plot_freq, verbose=verbose)
        
        
    def _on_rollout_end(self):
        # Zero advantage for clipped actions
        buffer = self.model.rollout_buffer
        
        actions = buffer.actions
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        advantages = buffer.advantages
        if isinstance(advantages, torch.Tensor):
            advantages = advantages.cpu().numpy()
            
        low = self.model.action_space.low
        high = self.model.action_space.high

        # Detect clipped actions
        clipped = (actions < low) | (actions > high)
        clipped = clipped.any(axis=1)  # (n_steps,) boolean
        
        n_clipped = clipped.sum()
        if n_clipped > 0:
            print(f"[Callback] Zeroing {n_clipped} clipped advantages")

        advantages[clipped] = 0.0
        # Convert back to tensor
        buffer.advantages = torch.tensor(advantages, device=self.model.device)


def compute_pyfolio_stats(returns):
    stats = pf.timeseries.perf_stats(returns)
    return stats

def run_model_on_env(model, env, horizon=252):
    obs, _ = env.reset()
    rewards = []
    for t in range(horizon):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        if done or truncated:
            break
    return pd.Series(rewards, index=pd.RangeIndex(len(rewards)))


def evaluate_models_on_same_envs(models, env_fns, horizon=252):
    """
    Evaluate multiple models on the SAME set of envs, using env_fns list of callables returning fixed ARCH(1) envs (same sequence per model)
    """
    results = {}
    for name, model in models.items():
        run_stats = []
        run_returns = []
        for env_fn in env_fns:
            env = env_fn()  # fresh env with fixed seed/path
            returns = run_model_on_env(model, env, horizon)
            run_returns.append(returns)
            run_stats.append(compute_pyfolio_stats(returns))

        stats_df = pd.DataFrame(run_stats)
        avg_stats = stats_df.mean()
        avg_stats["Max drawdown"] = stats_df["Max drawdown"].min()  # worst drawdown
        results[name] = (avg_stats, stats_df, run_returns)
    return results

def evaluate_models_on_same_envs2(models, env_fns, horizon=252):
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        # fresh DummyVecEnv for this model (so it gets the same paths)
        vec_env = DummyVecEnv(env_fns)

        run_returns = run_model_on_env(model, vec_env, horizon)
        run_stats = [compute_pyfolio_stats(r) for r in run_returns]

        stats_df = pd.DataFrame(run_stats)
        avg_stats = stats_df.mean()
        avg_stats["Max drawdown"] = stats_df["Max drawdown"].min()
        results[name] = (avg_stats, stats_df, run_returns)

        vec_env.close()

    return results


def run_model_on_env2(model, vec_env, horizon=252):
    obs = vec_env.reset()
    rewards = [[] for _ in range(vec_env.num_envs)]
    for t in range(horizon):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, infos = vec_env.step(action)
        # reward is vectorized: shape (n_envs,)
        for i, r in enumerate(reward):
            rewards[i].append(r)
        if done.any() or truncated.any():
            break
    # Convert each env’s rewards to a pd.Series
    return [pd.Series(r, index=pd.RangeIndex(len(r))) for r in rewards]