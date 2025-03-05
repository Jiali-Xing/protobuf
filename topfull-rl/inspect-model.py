import os, sys
import numpy as np
import torch
from stable_baselines3 import PPO

# sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'topfull-rl'))

from rl_simulator import MicroserviceDAGEnv  # Import the environment

# Load the trained model
# model_path = "./checkpoints/ppo_final_model.zip"
model_path = "./checkpoints/ppo_model_2370000_steps"
model = PPO.load(model_path)

# Define a function to evaluate the model and collect performance data
def evaluate_model(model, num_episodes=100):
    env = MicroserviceDAGEnv()  # Use the imported environment
    all_rewards = []
    all_goodput = []
    all_latency_penalties = []
        
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        total_goodput = 0
        max_latency = 0

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)
            total_reward += reward
            total_goodput += obs[1] # Assuming goodput is in obs[1] and obs[3]
            max_latency += obs[0]

            if terminated:
                break

        all_rewards.append(total_reward)
        all_goodput.append(total_goodput)
        all_latency_penalties.append(max_latency)

    return all_rewards, all_goodput, all_latency_penalties

# Run the evaluation
num_episodes = 0
rewards, goodput, latency_penalties = evaluate_model(model, num_episodes=num_episodes)

# Plot the performance trends (as before)
import matplotlib.pyplot as plt

episodes = np.arange(1, num_episodes + 1)

plt.figure(figsize=(14, 8))

# Plot rewards
plt.subplot(3, 1, 1)
plt.plot(episodes, rewards, label='Total Reward', color='blue')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward over Episodes')
plt.legend()

# Plot goodput
plt.subplot(3, 1, 2)
plt.plot(episodes, goodput, label='Total Goodput', color='green')
plt.xlabel('Episode')
plt.ylabel('Goodput')
plt.title('Goodput over Episodes')
plt.legend()

# Plot latency penalties
plt.subplot(3, 1, 3)
plt.plot(episodes, latency_penalties, label='Max Latency', color='red')
plt.xlabel('Episode')
plt.ylabel('Max Latency Among DAGs')
plt.title('Latency over Episodes')
plt.legend()

plt.tight_layout()
plt.savefig("performance_trends.png")
print("Performance trends saved to performance_trends.png")
plt.show()
