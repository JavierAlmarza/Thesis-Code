import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from scipy.io import loadmat
import pyfolio as pf
import pandas as pd
import modFin2 as mF2

import gymnasium as gym
from gymnasium import spaces
import modFin as mF
import matplotlib.pyplot as plt
import torch
import sys
import ctypes
import time
import yfinance as yf


sector_tickers = [
    "^SP500-50",  # Communication Services cyan
    "^SP500-25",  # Consumer Discretionary aquamarine
    "^SP500-30",  # Consumer Staples light green
    "^GSPE",  # Energy brown
    "^SP500-40",  # Financials pink
    "^SP500-35",  # Health Care orange
    "^SP500-20",  # Industrials violet
    "^SP500-45",  # Information Technology dark grayish brown
    "^SP500-15",  # Materials dark green
    "^SP500-60",  # Real Estate gray
    "^SP500-55"   # Utilities lavender
]

start = "2005-01-01"
end = "2024-01-01"

# Now auto_adjust=True is default, so this returns already adjusted Close
data = yf.download(sector_tickers,  start=start, end=end, interval="1d")["Close"]

# Compute daily returns
returns = data.pct_change().dropna()
cumulative_returns = (1 + returns).cumprod()

R = returns.to_numpy()
R = R.T


vix = yf.download("^VIX", start=start, end=end, interval="1d")["Close"]
sp500 = yf.download("^GSPC", start=start, end=end, interval="1d")["Close"]
sp500_returns = sp500.pct_change()
vol20 = sp500_returns.rolling(window=20).std() * np.sqrt(252)
vol20a = vol20.to_numpy()
vol60 = sp500_returns.rolling(window=60).std() * np.sqrt(252)
vol60a = vol60.to_numpy()
vix1  = vix.to_numpy()
vol2060 = vol20a/vol60a

episode_length = 252*4
T = 252*5

n_envs = 16
def make_env_Train(): # Vectorize environment for parallel PPO training
    return mF2.MarketReplayEnv(R[:,:T],vol20a[0:T],vol2060[0:T],vix1[0:T], episode_length=episode_length)
    
marketEnvTraining = make_vec_env(make_env_Train, n_envs = n_envs)



N_agents = 5

policy = 'MlpPolicy'
use_sde = False
gamma1=1
learning_rate=3e-4
#learning_rate=mF.linear_lr_schedule

ppo_agents = []
for k in range(N_agents):
    model = PPO(policy, marketEnvTraining , learning_rate=learning_rate, use_sde=use_sde, policy_kwargs=dict(net_arch=[64, 64],log_std_init=-1.0), ent_coef=0.00, gamma=gamma1, gae_lambda=0.9, clip_range=0.25, n_epochs=10,  batch_size=125, n_steps=250, verbose=1)
    ppo_agents.append(model)
 
total_timesteps= 2500

for k in range(len(ppo_agents)):
    ppo_agents[k].learn(total_timesteps=total_timesteps)
    
    
window = 60
val_episode_length = 252  # one trading year

# Build the sliced arrays for validation:
R_val = R[:, T - window : T + val_episode_length]            # shape (11, window + episode)
vol20_val = vol20a[T - window : T + val_episode_length]
vol2060_val = vol2060[T - window : T + val_episode_length]
vix_val = vix1[T - window : T + val_episode_length]

def make_env_Val(): # Vectorize environment for parallel PPO training
    return mF2.DeterministicValidationEnv(R_val, vol20_val, vol2060_val, vix_val, eta=0.01, window=window, episode_length=val_episode_length)
    
val_env = make_vec_env(make_env_Val, n_envs = n_envs)

print('EVALUATION STARTSSS')                                     
best_idx, results = mF2.evaluate_agents_sharpe(ppo_agents, R, vol20a, vol2060, vix1, T,
                                           window=60, val_episode_length=252, verbose=True)

best_agent = ppo_agents[best_idx]
best_sharpe = results[best_idx]['sharpe']
best_daily_returns = results[best_idx]['daily_returns']

print('All good!')