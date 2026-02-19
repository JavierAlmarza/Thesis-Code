import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from scipy.io import loadmat
import pyfolio as pf
import pandas as pd


import gymnasium as gym
from gymnasium import spaces
import modFin as mF
import matplotlib.pyplot as plt
import torch
import sys
import ctypes
import time

torch.set_num_threads(16)

# Prevent sleep and display turning off
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

# Prevent screen and system sleep
if sys.platform == "win32":
    # Prevent sleep on Windows
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
	

alpha0 = 0.0001
alpha1 = 0.25
N = 300
T = 1008
gamma1=1

r = 0.00012
k = 0.15
q = 0
total_timesteps = 10000
n_envs = 16
include_action = False
include_vol = False
UsesVecNormalize=False
#policy = mF.CustomPolicy
#policy = mF.SquashedGaussianPolicy
policy = 'MlpPolicy'
use_sde = False
#learning_rate=3e-4
learning_rate=mF.linear_lr_schedule
UsesSCA = False

# Generate synthetic data
R = mF.generate_arch1(N, T, alpha0, alpha1)
#R = mF.generate_gaussian(N,T,0,alpha0/(1-alpha1))
idx = np.random.randint(N)
chosen_row = R[idx, :]
R_Replay = np.tile(chosen_row, (N, 1))

data = loadmat('BaryDataMatrix.mat')
R_Bary = data['xsimData']

# Initialize Environment

def make_env_Bary(): # Vectorize environment for parallel PPO training
    return mF.PortfolioEnv(R_Bary, r=r, k=k, q=q, include_action=include_action, include_vol=include_vol)

def make_env_Replay(): # Vectorize environment for parallel PPO training
    return mF.PortfolioEnv(R_Replay, r=r, k=k, q=q, include_action=include_action, include_vol=include_vol)

env_Bary = make_vec_env(make_env_Bary, n_envs=n_envs)
env_Replay = make_vec_env(make_env_Replay, n_envs=n_envs)


# Create models
modelFin_Replay = PPO(policy, env_Replay, learning_rate=learning_rate, use_sde=use_sde, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0), ent_coef=0.00, gamma=gamma1, gae_lambda=0.9, clip_range=0.25, n_epochs=10,  batch_size=125, n_steps=250, verbose=1)
modelFin_Bary = PPO(policy, env_Bary, learning_rate=learning_rate, use_sde=use_sde, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0), ent_coef=0.00, gamma=gamma1, gae_lambda=0.9, clip_range=0.25, n_epochs=10,  batch_size=125, n_steps=250, verbose=1) 

# Learn

callback_Bary = mF.FinModelCallback(plot_freq=375, saveName = "trainingPlotBary.png")
modelFin_Bary.learn(total_timesteps=total_timesteps, callback = callback_Bary)  # Training

callback_Replay = mF.FinModelCallback(plot_freq=375, saveName = "trainingPlotReplay.png")
modelFin_Replay.learn(total_timesteps=total_timesteps, callback = callback_Replay)  # Training


# Save model
modelFin_Replay.save("modFinReplay")
modelFin_Bary.save("modFinBary")



# Generate synthetic data for backtest
RBtests = mF.generate_arch1(10, 750, alpha0, alpha1)

# Initialize Environment for backtest

def make_env_test(path):
    path_2d = path.reshape(1, -1)   # shape (1, T)
    return mF.PortfolioEnv(path_2d, r=r, k=k, q=q, include_action=include_action,include_vol=include_vol)


seeds = np.arange(10)
env_fns = [lambda path=RBtests[i]: make_env_test(path) for i in range(RBtests.shape[0])]

# Evaluate Both Models Fairly

models = {"Replay Model": modelFin_Replay, "Barycenter Model": modelFin_Bary}
results = mF.evaluate_models_on_same_envs(models, env_fns, horizon=749)

# Show Results

for name, (avg_stats, stats_df, run_returns) in results.items():
    print(f"\n=== {name} Results (avg over runs) ===")
    print(avg_stats)



