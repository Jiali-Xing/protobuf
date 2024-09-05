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
    def __init__(self, app_name, apis, max_steps=50, penalty_coefficient=1.0, slo=100, entry_point="nginx"):
        super(RealAppEnv, self).__init__()
        self.app_name = app_name
        self.apis = apis  # List of APIs in the cluster
        self.max_steps = max_steps
        self.penalty_coefficient = penalty_coefficient  # Penalty coefficient (ρ)
        self.slo = slo  # Service Level Objective for latency
        
        # Define priority map
        self.priority_map = {
            "compose": 1,
            "home-timeline": 2,
            "user-timeline": 3,
            "S_149998854": 2,
            "S_161142529": 3,
            "S_102000854": 1,
            "hotels-http": 3,
            "reservation-http": 1,
            "user-http": 2,
            "recommendations-http": 4,
            "motivate-set": 1,
            "motivate-get": 2,
            "search-hotel": 1,
            "store-hotel": 2,
            "reserve-hotel": 3,
        }

        # Get the server address from the Kubernetes service
        self.server_addresses = {}
        for api in apis:
            server_address = get_server_address(entry_point)
            if server_address is None:
                raise ValueError(f"Error retrieving server address for {api}")
            self.server_addresses[api] = server_address
            print(f"Server address for {api}: {self.server_addresses[api]}")

        self.rate_limits = {api: 3000 for api in apis}  # Initial rate limit for each API
        self.prev_goodput = {api: None for api in apis}  # To store previous goodput for each API
        self.current_latency = {api: 0 for api in apis}
        self.current_step = 0

        # Observation space: [aggregate_goodput_ratio, max_latency]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

        # Action space: single continuous action to adjust the rate limits
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.rate_limits = {api: 3000 for api in self.apis}  # Reset rate limits
        return self._get_observation(), {}

    def _get_observation(self):
        aggregate_goodput = 0
        aggregate_rate_limit = 0
        max_latency = 0

        for api in self.apis:
            # GET metrics from the Go server for each API
            params = {"method": api}
            response = requests.get(f"{self.server_addresses[api]}/metrics", params=params)
            metrics = response.json()

            # Assuming the metrics are returned as key-value pairs with API as key
            total_latency = metrics["latency"]
            total_goodput = metrics["goodput"]
            
            # Aggregate goodput and rate limits
            aggregate_goodput += total_goodput
            aggregate_rate_limit += self.rate_limits[api]
            
            # Track the maximum latency
            max_latency = max(max_latency, total_latency)

        goodput_ratio = aggregate_goodput / aggregate_rate_limit if aggregate_rate_limit > 0 else 0
        
        return np.array([goodput_ratio, max_latency], dtype=np.float32)

    def step(self, action):
        start_time = time.time()  # Record start time

        # below is Algorithm 1 from the paper TopFull 
        action_rl = action[0]
        if action_rl > 0:
            # Highest priority API (lowest priority value)
            api = min(self.apis, key=lambda api: self.priority_map.get(api, float('inf')))
        else:
            # Lowest priority API (highest priority value)
            api = max(self.apis, key=lambda api: self.priority_map.get(api, float('inf')))

        # Adjust rate limits based on action and priority
        # for api in targets:
        self.rate_limits[api] = max(1000, min(9000, self.rate_limits[api] * (1 + action_rl)))
        print(f"New rate limit for {api}: {self.rate_limits[api]}")
        # SET the new rate limit on the Go server
        params = {'method': api}
        data = {'rate_limit': self.rate_limits[api]}
        requests.post(f"{self.server_addresses[api]}/set_rate", params=params, json=data)

        observation = self._get_observation()
        goodput_ratio, max_latency = observation

        # Calculate aggregate goodput
        total_goodput = goodput_ratio * sum(self.rate_limits.values())

        # Calculate the change in goodput (ΔGoodput)
        delta_goodput = total_goodput - sum(self.prev_goodput.values()) if all(self.prev_goodput.values()) else 0

        # Update previous goodput
        for api in self.apis:
            self.prev_goodput[api] = goodput_ratio * self.rate_limits[api]

        # Calculate the penalty for latency exceeding the SLO
        latency_penalty = max(0, max_latency - self.slo)

        # Reward calculation
        reward = delta_goodput - self.penalty_coefficient * latency_penalty

        self.current_step += 1
        done = self.current_step >= self.max_steps

        elapsed_time = time.time() - start_time
        time.sleep(max(0, 1 - elapsed_time))  # Enforce 1-second interval
        
        return observation, reward, done, False, {}  # Return False for 'truncated' as this example doesn't use truncation.

    def close(self):
        pass

# Main function to run the RL agent across clusters
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rl_multicluster.py <application_name>")
        sys.exit(1)

    methods = sys.argv[1]
    print(f"Applying model on the {methods} interface")

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
    else:
        clusters = {
            "default": [methods]
        }

    # Loop through each cluster and apply the RL model
    for cluster_name, apis in clusters.items():
        if 'social' in methods:
            entry_point = "nginx" 
            app_name = "social"
        elif 'hotel' in methods:
            entry_point = "frontend"
            app_name = "hotel"
        else:
            entry_point = "nginx-web-server"

        env = RealAppEnv(app_name=methods, apis=apis, entry_point=entry_point)

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