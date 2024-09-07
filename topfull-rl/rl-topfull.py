import requests, time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os, subprocess, re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ghz-results'))
from slo import get_slo, get_sustainable_load


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
    def __init__(self, app_name, apis, max_steps=50, penalty_coefficient=1.0, entry_point="nginx"):
        super(RealAppEnv, self).__init__()
        self.app_name = app_name
        self.max_steps = max_steps
        self.penalty_coefficient = penalty_coefficient  # Penalty coefficient (ρ)

        self.apis = apis  # List of APIs in the cluster

        # Priority map for APIs
        self.priority_map = {
            "compose": 1,
            "home-timeline": 2,
            "user-timeline": 3,
            "S_102000854": 1,
            "S_149998854": 2,
            "S_161142529": 3,
            "motivate-set": 1,
            "motivate-get": 2,
            "search-hotel": 1,
            "reserve-hotel": 2
        }

        # Get the server address from the Kubernetes service
        server_address = get_server_address(entry_point)
        if server_address is None:
            raise ValueError("Error retrieving server address")
        
        self.server_address = server_address
        print(f"Server address: {self.server_address}")

        # updated for multiple APIs
        self.rate_limits = {api: 3000 for api in self.apis}
        self.prev_total_goodput = None  # To store previous goodput for ΔGoodput calculation
        self.current_latencies = {api: 0 for api in self.apis}
        self.current_step = 0

        # Observation space: [total_latency, total_goodput]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

        # Action space: single continuous action to adjust the rate limit
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.rate_limit = {api: 3000 for api in self.apis}  # Reset rate limit
        # You can set a random seed here for the environment
        if seed is not None:
            np.random.seed(seed)
        return self._get_observation(), {}

    def _get_observation(self):
        
        aggregate_goodput = 0
        aggregate_rate_limit = 0
        max_latency = 0
        
        for api in self.apis:
            params = {"method": api}
            # GET metrics from the Go server
            response = requests.get(f"{self.server_address}/metrics", params=params)
            metrics = response.json()
            total_latency = metrics["latency"]
            total_goodput = metrics["goodput"]
            # Calculate the ratio of goodput to the current rate limit

            aggregate_goodput += total_goodput
            aggregate_rate_limit += self.rate_limit[api]
            max_latency = max(max_latency, total_latency)

        goodput_ratio = aggregate_goodput / aggregate_rate_limit if aggregate_rate_limit > 0 else 0
        
        # The observation now includes:
        # 1. Ratio of goodput to the current rate limit
        # 2. The maximum latency across candidate APIs (already max_latency)
        return np.array([goodput_ratio, max_latency], dtype=np.float32)

    def step(self, action):
        start_time = time.time()  # Record start time

        # Apply Algorithm 1 from the paper
        action_rl = action[0]
        if action_rl > 0:
            # Highest priority API (lowest priority value)
            # api = min(self.apis, key=lambda api: self.priority_map.get(api, float('inf')))
            sorted_apis = sorted(self.apis, key=lambda api: self.priority_map.get(api, float('inf')))
        else:
            # Lowest priority API (highest priority value)
            # api = max(self.apis, key=lambda api: self.priority_map.get(api, 0))
            sorted_apis = sorted(self.apis, key=lambda api: -self.priority_map.get(api, float('inf')))

        # Loop through the sorted APIs and apply the action to the first valid one
        for api in sorted_apis:
            sustainable_load = get_sustainable_load(api)
            upper_bound = 2 * sustainable_load
            lower_bound = sustainable_load / 5

            current_rate_limit = self.rate_limits[api]

            # If we're increasing the rate and the rate limit is below the upper bound, apply the action
            if action_rl > 0 and current_rate_limit < upper_bound:
                self.rate_limits[api] = min(upper_bound, current_rate_limit * (1 + action_rl))
                break  # Apply the action to the first valid API and exit the loop

            # If we're decreasing the rate and the rate limit is above the lower bound, apply the action
            elif action_rl < 0 and current_rate_limit > lower_bound:
                self.rate_limits[api] = max(lower_bound, current_rate_limit * (1 + action_rl))
                break  # Apply the action to the first valid API and exit the loop

        
        # SET the new rate limit on the Go server for the selected API
        params = {'method': api}
        data = {'rate_limit': self.rate_limits[api]}
        requests.post(f"{self.server_address}/set_rate", params=params, json=data)
        
        observation = self._get_observation()

        # Extract goodput_ratio and max_latency from the observation
        goodput_ratio, max_latency = observation

        # For total_goodput, compute it based on the goodput_ratio and current total rate limit
        total_goodput = goodput_ratio * sum(self.rate_limits.values())

        # Calculate the change in goodput (ΔGoodput)
        delta_goodput = total_goodput - self.prev_total_goodput if self.prev_total_goodput is not None else 0

        # Update previous goodput
        self.prev_total_goodput = total_goodput

        # Calculate the penalty for latency exceeding the SLO
        # let's find the max SLO for apis
        latency_penalty = max(0, max_latency - max(get_slo(api) for api in self.apis))

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
    if len(sys.argv) != 3:
        print("Usage: python rl-topfull.py <method> <entry_point>")
        sys.exit(1)

    methods = sys.argv[1]
    entry_point = sys.argv[2]
    print(f"Applying model on the {methods} application")

    # Define clusters and APIs
    if methods == "all-social":
        clusters = {
            "cluster1": ["compose", "home-timeline"],
            "cluster2": ["user-timeline"]
        }
    elif methods == "both-hotel":
        clusters = {
            "cluster1": ["search-hotel", "reserve-hotel"]
        }
    elif methods == "both-motivate":
        clusters = {
            "cluster1": ["motivate-set", "motivate-get"]
        }
    elif methods == "all-alibaba":
        clusters = {
            "cluster1": ["S_102000854"],
            "cluster2": ["S_149998854"],
            "cluster3": ["S_161142529"]
        }
    else:
        clusters = {
            "default": [methods]
        }
    
    # Loop through each cluster and apply the RL model
    for cluster_name, apis in clusters.items():
        # Set entry_point and app_name based on the method
        if 'social' in methods or 'compose' in methods or 'timeline' in methods:
            app_name = "social"
        elif 'hotel' in methods:
            app_name = "hotel"
        elif 'motivate' in methods:
            app_name = "motivate"
        elif 'alibaba' in methods or 'S_' in methods:
            app_name = "alibaba"
        else:
            app_name = methods  # Default case uses the method as app_name

            
        # Pass the 'apis' argument to the RealAppEnv class
        env = RealAppEnv(app_name=app_name, apis=apis, entry_point=entry_point)

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
