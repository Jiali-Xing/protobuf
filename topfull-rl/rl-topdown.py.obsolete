import requests, time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os, subprocess, re
import sys


def get_server_address(entry_point, port=8082):
    try:
        # Get the Cluster IP of the service
        service_ip = subprocess.check_output(
            f"kubectl get service {entry_point} -o=jsonpath='{{.spec.clusterIP}}'", shell=True
        ).decode('utf-8').strip()

        # Get the NodePort (optional, depending on your use case)
        node_port = subprocess.check_output(
            f"kubectl get service {entry_point} -o=jsonpath='{{.spec.ports[0].nodePort}}'", shell=True
        ).decode('utf-8').strip()

        # Construct the service URL
        server_address = f"http://{service_ip}:{port}"
        return server_address
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving server address: {e}")
        return None
    

class RealAppEnv(gym.Env):
    def __init__(self, app_name, max_steps=50, penalty_coefficient=1.0, slo=100, entry_point="nginx"):
        super(RealAppEnv, self).__init__()
        self.app_name = app_name
        self.max_steps = max_steps
        self.penalty_coefficient = penalty_coefficient  # Penalty coefficient (ρ)
        self.slo = slo  # Service Level Objective for latency
        
        # Get the server address from the Kubernetes service
        server_address = get_server_address(entry_point)
        if server_address is None:
            raise ValueError("Error retrieving server address")
        
        self.server_address = server_address
        print(f"Server address: {self.server_address}")

        self.rate_limit = 3000  # Initial rate limit
        self.prev_goodput = None  # To store previous goodput for ΔGoodput calculation
        self.current_latency = 0
        self.current_step = 0

        # Observation space: [total_latency, total_goodput]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

        # Action space: single continuous action to adjust the rate limit
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.rate_limit = 3000  # Reset rate limit
        # You can set a random seed here for the environment
        if seed is not None:
            np.random.seed(seed)
        return self._get_observation(), {}

    def _get_observation(self):
        # GET metrics from the Go server
        response = requests.get(f"{self.server_address}/metrics")
        metrics = response.json()
        total_latency = metrics["latency"]
        total_goodput = metrics["goodput"]
        # Calculate the ratio of goodput to the current rate limit
        total_rate_limit = self.rate_limit  # Aggregate sum of current rate limits
        goodput_ratio = total_goodput / total_rate_limit if total_rate_limit > 0 else 0
        
        # The observation now includes:
        # 1. Ratio of goodput to the current rate limit
        # 2. The maximum latency across candidate APIs (already max_latency)
        return np.array([goodput_ratio, total_latency], dtype=np.float32)

    def step(self, action):
        start_time = time.time()  # Record start time

        self.rate_adjustment = action[0]
        self.rate_limit = max(1000, min(9000, self.rate_limit * (1 + self.rate_adjustment)))

        # SET the new rate limit on the Go server
        requests.post(f"{self.server_address}/set_rate", json={"rate_limit": self.rate_limit})

        observation = self._get_observation()

        # Extract goodput_ratio and max_latency from the observation
        goodput_ratio, total_latency = observation

        # For total_goodput, compute it based on the goodput_ratio and current total rate limit
        total_rate_limit = self.rate_limit
        total_goodput = goodput_ratio * total_rate_limit

        # Calculate the change in goodput (ΔGoodput)
        if self.prev_goodput is None:
            delta_goodput = 0  # No change at the start
        else:
            delta_goodput = total_goodput - self.prev_goodput

        self.prev_goodput = total_goodput
        self.current_latency = total_latency

        # Calculate the penalty for latency exceeding the SLO
        latency_penalty = max(0, total_latency - self.slo)

        # Reward calculation
        reward = delta_goodput - self.penalty_coefficient * latency_penalty

        self.current_step += 1
        done = self.current_step >= self.max_steps

        elapsed_time = time.time() - start_time
        time.sleep(max(0, 1 - elapsed_time))  # Enforce 1-second interval
        
        return observation, reward, done, False, {}  # Return False for 'truncated' as this example doesn't use truncation.

    def close(self):
        pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python apply_model.py <application_name>")
        sys.exit(1)

    app_name = sys.argv[1]
    print(f"Applying model on the {app_name} application")

    entry_point = "nginx" if app_name == "social" else "frontend"
    env = RealAppEnv(app_name=app_name, entry_point=entry_point)

    # Load the final trained model
    final_model_path = f"{app_name}_checkpoints/{app_name}_final_model.zip"
    model = PPO.load(final_model_path, env=env)
    print(f"Loaded model from {final_model_path}")

    # Apply the trained model in the real environment
    obs, _ = env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
