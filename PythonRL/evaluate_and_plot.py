import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import modFin as mF
#from FinanceModel import PortfolioEnv  # Replace with the actual filename (without .py)
#from FinanceModel import generate_arch1
import gymnasium as gym


# Generate data
alpha0 = 0.01
alpha1 = 0.25
N = 300
T = 750

R = mF.generate_arch1(N, T, alpha0, alpha1)

# Create environment
env = mF.PortfolioEnv(R=R, r=0.0012, k=0.1, q=4)

# Load environment
model = PPO.load("modFin")  

# Evaluate trained model
n_eval_episodes = 50
ppo_rewards = []
ppo_actions = []

for episode in range(n_eval_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        ppo_actions.append(action)
    ppo_rewards.append(total_reward)

# Evaluate random policy
env_random = mF.PortfolioEnv(R=R, r=0.0012, k=0.1, q=4)
random_rewards = []

for episode in range(n_eval_episodes):
    obs, _ = env_random.reset()
    done = False
    total_reward = 0
    while not done:
        action = env_random.action_space.sample()
        obs, reward, terminated, truncated, info = env_random.step(action)
        done = terminated or truncated
        total_reward += reward
    random_rewards.append(total_reward)

# Print averages

avg_PPO = sum(ppo_rewards)/len(ppo_rewards)
avg_random = sum(random_rewards)/len(random_rewards)

print('PPO average reward is')
print(avg_PPO)
print('Random average reward is')
print(avg_random)

# Plot results

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

ax1.plot(ppo_rewards, label="PPO Reward")
ax1.plot(random_rewards, label="Random Reward")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_title("PPO vs Random Policy")
ax1.legend()
ax1.grid(True)

ax2.plot(ppo_actions, label="PPO Action", color="red")
ax2.set_xlabel("Step")
ax2.set_ylabel("Action")
ax2.legend()
ax2.grid(True)

plt.show()





