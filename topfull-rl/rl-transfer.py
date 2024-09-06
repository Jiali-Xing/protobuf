import requests, time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure
import os, sys, re

class RealAppEnv(gym.Env):
    def __init__(self, app_name, apis, max_steps=50, penalty_coefficient=1.0, slo=100, entry_point="nginx"):
        super(RealAppEnv, self).__init__()
        self.app_name = app_name
        self.apis = apis  # List of APIs in the cluster
        self.max_steps = max_steps
        self.penalty_coefficient = penalty_coefficient  # Penalty coefficient (ρ)
        self.slo = slo  # Service Level Objective for latency

        self.priority_map = {
				"S_149998854":          1,
				"S_161142529":          2,
				"S_102000854":          3,
                "motivate-set":         1,
                "motivate-get":         2,
			},

        # Get the server address from the user
        server_address = input("Enter the server address: ")
        server_address = f'http://{server_address}'

        if server_address is None:
            raise ValueError("Error retrieving server address")
        
        self.server_address = server_address
        print(f"Server address: {self.server_address}")

        # Initialize rate limits for each API
        self.rate_limits = {api: 3000 for api in apis}
        self.prev_goodput = {api: None for api in apis}  # To store previous goodput for ΔGoodput calculation
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
            response = requests.get(f"{self.server_address}/metrics", params=params)
            metrics = response.json()

            total_latency = metrics["latency"]
            total_goodput = metrics["goodput"]
            
            # Aggregate goodput and rate limits
            aggregate_goodput += total_goodput
            aggregate_rate_limit += self.rate_limits[api]
            
            # Track the maximum latency across APIs
            max_latency = max(max_latency, total_latency)

        # Calculate the ratio of goodput to the current rate limits
        goodput_ratio = aggregate_goodput / aggregate_rate_limit if aggregate_rate_limit > 0 else 0
        
        # Return observation with goodput ratio and max latency
        return np.array([goodput_ratio, max_latency], dtype=np.float32)

    def step(self, action):
        start_time = time.time()  # Record start time

        action_rl = action[0]
        if action_rl > 0:
            # Highest priority API (using hardcoded priorities)
            api = min(self.apis, key=lambda api: self.priority_map.get(api, float('inf')))
        else:
            # Lowest priority API
            api = max(self.apis, key=lambda api: self.priority_map.get(api, float('inf')))

        # Adjust rate limits based on action and priority
        self.rate_limits[api] = max(1000, min(9000, self.rate_limits[api] * (1 + action_rl)))
        
        # Set the new rate limit on the server
        params = {'method': api}
        data = {'rate_limit': self.rate_limits[api]}
        requests.post(f"{self.server_address}/set_rate", params=params, json=data)

        observation = self._get_observation()
        goodput_ratio, max_latency = observation

        # Calculate aggregate goodput
        total_goodput = goodput_ratio * sum(self.rate_limits.values())

        # Calculate the change in goodput (ΔGoodput)
        delta_goodput = total_goodput - sum(self.prev_goodput.values()) if all(self.prev_goodput.values()) else 0

        # Update previous goodput for each API
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

class PrintCallback(BaseCallback):
    def __init__(self, check_freq, max_prints=50, verbose=1):
        super(PrintCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.max_prints = max_prints
        self.print_count = 0  # Initialize the print counter

    def _on_step(self):
        env = self.locals['env']
        if self.print_count < self.max_prints and self.n_calls % self.check_freq == 0:
            for i in range(env.num_envs):
                current_env = env.envs[i]
                print(f"Step {current_env.current_step}: Goodput = {current_env.prev_goodput}, Latency = {current_env.current_latency}")
                print(f"Rate Limit = {current_env.rate_limits}, action = {current_env.rate_adjustment}")
            self.print_count += 1
        return True

# Remaining parts (checkpoint, tensorboard, and training logic) stay the same, except for using the multi-API env

class CustomTensorBoardCallback(BaseCallback):
    def __init__(self, log_dir='./logs/', log_freq=1000, verbose=0):
        super(CustomTensorBoardCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir)
        self.n_calls = 0
        self.log_freq = log_freq

    def _on_step(self):
        self.n_calls += 1
        if self.n_calls % self.log_freq == 0:
            env = self.locals['env']
            # Log custom metrics
            for i in range(env.num_envs):
                current_env = env.envs[i]
                step = self.n_calls
                # Log to TensorBoard
                self.writer.add_scalar('Goodput', current_env.prev_goodput, step)
                self.writer.add_scalar('Latency', current_env.current_latency, step)
                self.writer.add_scalar('Rate Limit', current_env.rate_limit, step)
                self.writer.add_scalar('Learning Rate', model.optimizer.param_groups[0]['lr'], step)
        return True

    def _on_training_end(self):
        self.writer.close()

class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix='', verbose=1):
        super(CustomCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            current_step = self.num_timesteps  # This is the total number of timesteps so far
            save_path = os.path.join(self.save_path, f"{self.name_prefix}_{current_step}_steps.zip")
            self.model.save(save_path)
            if self.verbose > 0:
                print(f"Saved model checkpoint at step {current_step} to {save_path}")
        return True
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python apply_model.py <application_name>")
        sys.exit(1)

    app_name = sys.argv[1]
    print(f"Training on the {app_name} application")

    if app_name == "motivate":
        apis = ["motivate-get", "motivate-set"]
    elif app_name == "alibaba":
        apis = ["S_102000854", "S_149998854", "S_161142529"]
    else:
        raise ValueError("Unknown application name. Choose 'motivate' or 'alibaba'.")

    entry_point = "nginx-web-server"
    env = RealAppEnv(app_name=app_name, apis=apis, entry_point=entry_point)

    # Set up checkpoint, tensorboard, and training similar to the previous approach
    checkpoint_dir = app_name + "_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CustomCheckpointCallback(save_freq=50, save_path=checkpoint_dir, name_prefix=app_name+"_ppo")

    eval_env = RealAppEnv(app_name=app_name, apis=apis, entry_point=entry_point)

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=checkpoint_dir, 
        log_path=app_name + "_logs",
        eval_freq=10000,  
        n_eval_episodes=5,  
        deterministic=True,
        render=False
    )

    print_callback = PrintCallback(check_freq=20, max_prints=200)
    custom_tb_callback = CustomTensorBoardCallback(log_dir=app_name + "_logs", log_freq=1000)

    callbacks = CallbackList([checkpoint_callback, eval_callback, print_callback, custom_tb_callback])

    new_logger = configure(app_name + "_logs", ["stdout", "csv", "tensorboard"])

    pre_trained_model = "checkpoints-19/pretrained_model_final.zip"
    checkpoints = sorted(
        os.listdir(checkpoint_dir),
        key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x))
    )
    
    if checkpoints:
        last_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"Loading model from checkpoint: {last_checkpoint}")
        model = PPO.load(last_checkpoint, env=env)
        timestep_str = re.findall(r'\d+', last_checkpoint)
        if timestep_str:
            completed_timesteps = int(timestep_str[-1])
        else:
            completed_timesteps = 0
    else:
        model = PPO.load(pre_trained_model, 
                         env=env, 
                         tensorboard_log=app_name + "_logs",
                         verbose=1)
        completed_timesteps = 0

    model.set_logger(new_logger)

    total_timesteps = 50 * 800  
    remaining_timesteps = total_timesteps - completed_timesteps

    if remaining_timesteps > 0:
        model.learn(total_timesteps=remaining_timesteps, callback=callbacks, tb_log_name=app_name)
    else:
        print("Training is already complete.")

    model.save(os.path.join(checkpoint_dir, app_name + "_final_model"))

    obs, _ = env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
