import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from scipy.io import loadmat

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

r = 0.0012
k = 0.15
q = 1
total_timesteps = 350000
n_envs = 16
include_action = True
include_vol = True
UsesVecNormalize=False
#policy = mF.CustomPolicy
#policy = mF.SquashedGaussianPolicy
policy = 'MlpPolicy'
use_sde = False
#learning_rate=3e-4
learning_rate=mF.linear_lr_schedule
UsesSCA = False

# Generate synthetic data

#R = mF.generate_arch1(N, T, alpha0, alpha1)
#R = mF.generate_gaussian(N,T,0,alpha0/(1-alpha1))

data = loadmat('BaryDataMatrix.mat')
R = data['xsimData']


# Initialize Environment 

def make_env(): # Vectorize environment for parallel PPO training
    return mF.PortfolioEnv(R, r=r, k=k, q=q, include_action=include_action, include_vol=include_vol)

env = make_vec_env(make_env, n_envs=n_envs)

if UsesVecNormalize:
    # Create vectorized env (CHECKK!!!)
    env = DummyVecEnv([lambda: env])
    # Wrap with VecNormalize (normalizes observations and rewards)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

# Create model

if UsesSCA:
    modelFin = SAC(policy, env, learning_rate=learning_rate, use_sde=use_sde, buffer_size=10000, ent_coef=0.00, gamma=gamma1, verbose=1)
else:
    modelFin = PPO(policy, env, device="cuda" , learning_rate=learning_rate, use_sde=use_sde, policy_kwargs=dict(net_arch=[64, 64],log_std_init=-1.0), ent_coef=0.00, gamma=gamma1, gae_lambda=0.9, clip_range=0.25, n_epochs=10,  batch_size=125, n_steps=250, verbose=1)

# Check log_std after initialization (PPO only)
if not UsesSCA:
    print("\nChecking if log_std is trainable right after model initialization:")
    print("log_std requires_grad:", modelFin.policy.log_std.requires_grad)
    
    # Check if log_std is in optimizer parameters
    log_std_in_optimizer = any(
    modelFin.policy.log_std is p for g in modelFin.policy.optimizer.param_groups for p in g['params'])
    print("log_std is in optimizer:", log_std_in_optimizer)

# Add Callback for plotting during learning

callback = mF.FinModelCallback(plot_freq=375)

#Learn
plt.ion() # turn on interactive plot
if not UsesSCA:
    modelFin.learn(total_timesteps=total_timesteps, callback=callback)  # Training
else:
    modelFin.learn(total_timesteps=total_timesteps)  # Training


# Check log_std again after learning
if not UsesSCA:
    print("\nChecking log_std after training:")
    print("Current log_std:", modelFin.policy.log_std.data)
    print("Current sigma:", modelFin.policy.log_std.exp().data)

# Save model
modelFin.save("modFin")

# Add iterations to check performance
ans = input('How many iterations do you want?')
T = int(ans)
reset_output = env.reset()
if isinstance(reset_output, tuple):
    obs, info = reset_output
else:
    obs = reset_output
    info = {}

if use_sde == True:
    modelFin.policy.reset_noise()
actions=np.zeros((2,T))
waction=np.zeros(T)
rewards=np.zeros(T)

print("obs type:", type(obs))
if isinstance(obs, np.ndarray):
    print("obs.shape:", obs.shape)
    print("obs.dtype:", obs.dtype)
else:
    print("obs is not a numpy array:", obs)

boolTest = True

if boolTest:
    R2 = mF.generate_arch1(1, T, alpha0, alpha1)
    def make_env2(): # Vectorize environment for parallel PPO training
        return mF.PortfolioEnv(R2, r=r, k=k, q=q, include_action=include_action, include_vol=include_vol)
        env = make_vec_env2(make_env, n_envs=1)

for i in range(T):
    a, _ = modelFin.predict(obs, deterministic=False)
    #print(a)
    #actions[0,i] = a[0,0]
    #waction[i] = np.exp(actions[0,i])/(np.exp(actions[0,i])+np.exp(actions[1,i]) ) 
    waction[i] = a[0]
    if hasattr(env, "num_envs"):
        obs, reward, done, info = env.step(a)
        terminated = False
        truncated = False
        if done.any():
           break
    else:
        obs, reward, terminated, truncated, info = env.step(a)
        done = False
        if terminated or truncated:
           break
    rewards[i] = (reward[0] if np.ndim(reward) > 0 else reward)

sumrw = np.mean(rewards)*1008
print('Avg reward is ',sumrw)

#for name, param in modelFin.policy.named_parameters():
#    if 'log_std' in name:
#        print(name, param.requires_grad)

#plt.figure()
#plt.plot(waction,linestyle='-', linewidth=0.3, color='blue')
#plt.title("Actions Passed to Environment")
#plt.xlabel("Timestep")
#plt.ylabel("Action")
#plt.savefig("testing_plot.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

ax1.plot(rewards)
ax1.set_ylabel('Reward')
ax1.set_title('Step Rewards')
#ax1.legend()

ax2.plot(waction, label='Actions', color='green')
ax2.set_ylabel('Action')
ax2.set_xlabel('Timestep')
ax2.set_title('Sampled Actions')
#ax2.legend()

fig.tight_layout()
fig.savefig("testing_plot.png")  # saves to current working directory


# Turn off sleep prevention
if sys.platform == "win32":
    # Prevent sleep on Windows
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
