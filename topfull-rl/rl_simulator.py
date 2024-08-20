import numpy as np
# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces

import networkx as nx
import random, os, re
import torch

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure


class MicroserviceNode:
    def __init__(self, name, load_capacity, base_latency):
        self.name = name
        self.load_capacity = load_capacity
        self.base_latency = base_latency
        self.current_load = 0
        self.previous_load = 0  # To track the previous incoming rate
        self.previous_goodput = 0  # To track the previous goodput
        self.previous_latency = 0  # To track the previous latency
        self.overloaded = False


    def process_request(self, incoming_rate):
        self.previous_load = self.current_load
        self.current_load = incoming_rate

        if self.current_load > self.load_capacity:
            overload_factor = self.current_load / self.load_capacity
            self.overloaded = True

            if self.current_load > self.previous_load:
                # Overloaded and incoming rate is increasing
                latency = self.previous_latency + self.base_latency * overload_factor 
                goodput = max(0, self.previous_goodput - self.load_capacity * overload_factor * 0.1)
            else:
                # Overloaded but incoming rate is decreasing
                # Latency decreases as incoming rate decreases, goodput increases
                latency = self.previous_latency - self.base_latency * overload_factor * 0.1
                goodput = min(self.current_load, self.previous_goodput + self.load_capacity * overload_factor * 0.05)
        else:
            # Not overloaded
            self.overloaded = False
            overload_factor = 1

            latency = self.base_latency 
            goodput = incoming_rate

        # Ensure latency and goodput do not go below 0
        latency = max(0, latency + np.random.normal(0, self.base_latency * overload_factor * 0.1))
        goodput = max(0, goodput + np.random.normal(0, self.load_capacity * 0.001 * overload_factor))

        # but goodput cannot exceed the incoming rate
        goodput = min(goodput, self.current_load)

        # Save the current latency and goodput for the next step
        self.previous_latency = latency
        self.previous_goodput = goodput

        return latency, goodput


class EpisodeCheckpointCallback(BaseCallback):
    def __init__(self, save_freq_episodes: int, save_path: str, max_episodes: int, verbose: int = 0):
        super(EpisodeCheckpointCallback, self).__init__(verbose)
        self.save_freq_episodes = save_freq_episodes
        self.save_path = save_path
        self.max_episodes = max_episodes
        self.n_episodes = 0

    def _on_step(self) -> bool:
        # Check if the episode is done
        done = self.locals["dones"][0]
        if done:
            self.n_episodes += 1

            # Checkpoint the model every save_freq_episodes
            if self.n_episodes % self.save_freq_episodes == 0:
                save_file = os.path.join(self.save_path, f"model_{self.n_episodes}_episodes")
                self.model.save(save_file)
                if self.verbose > 0:
                    print(f"Checkpoint saved at {save_file}")

            # Stop training after max_episodes
            if self.n_episodes >= self.max_episodes:
                print(f"Training stopped after {self.max_episodes} episodes.")
                return False
        
        return True
    

class MicroserviceDAGEnv(gym.Env):
    def __init__(self, num_dags=1, max_nodes_per_dag=5, max_steps=10, penalty_coefficient=1.0, slo=100):
        super(MicroserviceDAGEnv, self).__init__()
        self.num_dags = num_dags
        self.max_nodes_per_dag = max_nodes_per_dag
        self.max_steps = max_steps
        self.current_step = 0

        self.dags = self._generate_dags()
        self.initial_user_request_rate = random.randint(5000, 6000)
        self.user_request_rate = self.initial_user_request_rate
        # self.incoming_rate is a list the size of the incoming rate equals the size of DAGs
        self.rate_limit = [self.initial_user_request_rate] * self.num_dags

        self.prev_goodput = None  # To store previous goodput for ΔGoodput calculation
        self.current_latency = 0

        self.penalty_coefficient = penalty_coefficient  # Penalty coefficient (ρ)
        self.slo = slo  # Service Level Objective for latency

        # Observation space: [total_latency, total_goodput]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

        # Action space: single continuous action to adjust the incoming rate
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)
        self.rate_adjustment = 0

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        return [seed]
    
    def _generate_dags(self):
        dags = []
        for _ in range(self.num_dags):
            num_nodes = random.randint(1, self.max_nodes_per_dag)
            dag = nx.DiGraph()
            
            for i in range(num_nodes):
                load_capacity = random.uniform(3500, 5000)  # Scale load capacity to be around 3000 RPS
                base_latency = random.uniform(1, 10)
                node = MicroserviceNode(name=f"node_{i}", load_capacity=load_capacity, base_latency=base_latency)
                dag.add_node(i, data=node)

            for i in range(1, num_nodes):
                parent_node = random.choice(list(dag.nodes)[:i])
                dag.add_edge(parent_node, i)
            
            dags.append(dag)
        return dags

    def reset(self, seed=None, **kwargs):
        # Set the seed if provided
        if seed is not None:
            self.seed(seed)

        self.dags = self._generate_dags()
        self.current_step = 0
        self.user_request_rate = self.initial_user_request_rate
        return self._get_observation()

    def _get_observation(self):
        # # Simulate the current DAG and return the observation
        total_goodput, max_latency = self.simulate(self.rate_limit)

        # Calculate the ratio of goodput to the current rate limit
        total_rate_limit = sum(self.rate_limit)  # Aggregate sum of current rate limits
        goodput_ratio = total_goodput / total_rate_limit if total_rate_limit > 0 else 0
        
        # The observation now includes:
        # 1. Ratio of goodput to the current rate limit
        # 2. The maximum latency across candidate APIs (already max_latency)
        return np.array([goodput_ratio, max_latency], dtype=np.float32)


    def step(self, action):
        self.rate_adjustment = action[0]
        # self.user_request_rate should be a random walk
        self.user_request_rate = max(3000, min(9000, self.user_request_rate + np.random.normal(0, 100)))

        # self.incoming_rate = min(self.user_request_rate, self.incoming_rate * (1 + self.rate_adjustment))
        # assume that each DAG is independent, and prioritize the DAG closer to index 0
        # aka, when action is positive, the first DAG will be prioritized to receive the increase in incoming rate
        # when action is negative, the last DAG will be de-prioritized to receive less incoming rate
        if self.rate_adjustment > 0:
            self.rate_limit[0] = self.rate_limit[0] * (1 + self.rate_adjustment)
        else:
            self.rate_limit[-1] = self.rate_limit[-1] * (1 + self.rate_adjustment)
        
        # clip the rate limit to be between 1000 and 8000
        self.rate_limit = [max(100, min(10000, rate)) for rate in self.rate_limit]

        observation = self._get_observation()
        # total_latency, total_goodput = observation

        # Extract goodput_ratio and max_latency from the observation
        goodput_ratio, total_latency = observation

        # For total_goodput, compute it based on the goodput_ratio and current total rate limit
        total_rate_limit = sum(self.rate_limit)
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

        # Use 'terminated' and 'truncated' flags to comply with SB3 expectations
        terminated = done  # Episode ended naturally
        truncated = False  # No truncation logic in this case

        return observation, reward, terminated, truncated, {}


    def simulate(self, rate_limit):
        total_goodput = 0
        max_latency = 0

        for i, dag in enumerate(self.dags):
            incoming_rate = min(self.user_request_rate, rate_limit[i])
            dag_latency = 0
            node_goodputs = []

            for node in nx.topological_sort(dag):
                node_data = dag.nodes[node]['data']
                latency, goodput = node_data.process_request(incoming_rate)
                dag_latency += latency
                node_goodputs.append(goodput)

            # Goodput for this DAG is the minimum goodput across all nodes
            final_goodput = min(node_goodputs)

            # Accumulate the total goodput across all DAGs
            total_goodput += final_goodput

            # Track the maximum latency across all DAGs
            if dag_latency > max_latency:
                max_latency = dag_latency

        return total_goodput, max_latency

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class PrintCallback(BaseCallback):
    def __init__(self, check_freq, max_prints=20, verbose=1):
        super(PrintCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.max_prints = max_prints
        self.print_count = 0  # Initialize the print counter


    def _on_step(self):
        # Access the environment
        env = self.locals['env']
        
        # Check if we're at the correct step interval and haven't exceeded max prints
        if self.print_count < self.max_prints and self.n_calls % self.check_freq == 0:
            # Assuming the environment is a vectorized environment
            for i in range(env.num_envs):
                # Access the environment instance if it's a DummyVecEnv or similar
                current_env = env.envs[i]
                
                # Print the current goodput, latency, etc.
                print(f"Step {current_env.current_step}: Goodput = {current_env.prev_goodput}, Latency = {current_env.current_latency}, User Request Rate = {current_env.user_request_rate}")
                print(f"Rate Limit = {current_env.rate_limit}, action = {current_env.rate_adjustment}")
            
            # Increment the print counter
            self.print_count += 1

        return True


def get_step_number(filename):
    match = re.search(r'ppo_model_(\d+)_steps\.zip', filename)
    if match:
        return int(match.group(1))
    return 0

from torch.utils.tensorboard import SummaryWriter

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
                self.writer.add_scalar('User Request Rate', current_env.user_request_rate, step)
                self.writer.add_scalar('Rate Limit', current_env.rate_limit[0], step)
                self.writer.add_scalar('Learning Rate', model.optimizer.param_groups[0]['lr'], step)
        return True

    def _on_training_end(self):
        self.writer.close()


# Example usage with PPO:
if __name__ == "__main__":

    # Set up the environment
    env = MicroserviceDAGEnv(num_dags=1, max_nodes_per_dag=4, max_steps=50, penalty_coefficient=10, slo=20)

    # Set up the evaluation environment (usually a copy of the training environment)
    eval_env = MicroserviceDAGEnv(num_dags=1, max_nodes_per_dag=4, max_steps=50, penalty_coefficient=10, slo=20)

    # Set up the checkpoint callback
    checkpoint_dir = "./checkpoints-19/"
    log_dir = "./logs-19/"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=checkpoint_dir, name_prefix="ppo_model")

    # Create the custom callback to checkpoint every 50 episodes and stop after 48,000 episodes
    checkpoint_callback = EpisodeCheckpointCallback(save_freq_episodes=50, save_path=checkpoint_dir, max_episodes=48000)

    # Find the last checkpoint by sorting based on the linux timestamp (not the get_step_number)
    checkpoints = sorted(
        os.listdir(checkpoint_dir),
        key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x))
    )
    
    if checkpoints:
        last_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"Loading model from checkpoint: {last_checkpoint}")
        model = PPO.load(last_checkpoint, env=env)
        # change the learning rate to default
        model.optimizer = torch.optim.Adam(model.policy.parameters(), lr=5e-5)
    else:
        print("No checkpoint found. Starting a new model.")
        # Define the PPO model with the specific parameters
        env = make_vec_env(MicroserviceDAGEnv, n_envs=5)  # Use 5 parallel environments
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=50,  # Steps in episode
            learning_rate=5e-5,  # Learning rate
            ent_coef=0.2,  # KL coefficient (used as entropy coefficient in SB3)
            target_kl=0.01,  # KL target
            batch_size=250,  # Minibatch size
            clip_range=0.3,  # PPO clip parameter
            verbose=1
        )
        model.optimizer = torch.optim.Adam(model.policy.parameters(), lr=5e-5)

    # Combine callbacks
    print_callback = PrintCallback(check_freq=20, max_prints=200)

    # Configure TensorBoard logger
    new_logger = configure(log_dir, ["tensorboard"])

    model.set_logger(new_logger)

    # # Add callbacks
    custom_tb_callback = CustomTensorBoardCallback(log_dir=log_dir)

    # Add the evaluation callback
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=checkpoint_dir, 
        log_path=log_dir,
        eval_freq=10000,  # Evaluate every 10,000 steps
        n_eval_episodes=5,  # Number of episodes to evaluate over
        deterministic=True,
        render=False
    )

    callbacks = CallbackList([checkpoint_callback, print_callback, eval_callback, custom_tb_callback])
    # Train the model
    model.learn(total_timesteps=1e10, callback=callbacks, tb_log_name="ppo_model")

    # Save the final model
    model.save(os.path.join(checkpoint_dir, "pretrained_model_final"))

    # Evaluate the model
    obs = env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        if done.any():
            obs = env.reset()
