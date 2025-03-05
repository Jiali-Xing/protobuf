import numpy as np
import networkx as nx
import random

class MicroserviceNode:
    def __init__(self, name, load_capacity, base_latency):
        self.name = name
        self.load_capacity = load_capacity
        self.base_latency = base_latency
        self.current_load = 0

    def process_request(self, incoming_rate):
        self.current_load = incoming_rate

        if self.current_load > self.load_capacity:
            overload_factor = self.current_load / self.load_capacity
            latency = self.base_latency * overload_factor + np.random.normal(0, self.base_latency * 0.5)
            goodput = max(0, self.load_capacity - np.random.normal(0, 0.1 * self.load_capacity))
        else:
            latency = self.base_latency + np.random.normal(0, self.base_latency * 0.5)
            goodput = incoming_rate

        return latency, goodput


class MicroserviceDAGSimulator:
    def __init__(self, num_dags=1, max_nodes_per_dag=5):
        self.num_dags = num_dags
        self.max_nodes_per_dag = max_nodes_per_dag
        self.dags = self._generate_dags()

    def _generate_dags(self):
        dags = []
        for _ in range(self.num_dags):
            num_nodes = random.randint(1, self.max_nodes_per_dag)
            dag = nx.DiGraph()
            
            # Add nodes to the DAG
            for i in range(num_nodes):
                load_capacity = random.uniform(10, 100)
                base_latency = random.uniform(1, 10)
                node = MicroserviceNode(name=f"node_{i}", load_capacity=load_capacity, base_latency=base_latency)
                dag.add_node(i, data=node)

            # Add edges between nodes
            for i in range(1, num_nodes):
                parent_node = random.choice(list(dag.nodes)[:i])
                dag.add_edge(parent_node, i)
            
            dags.append(dag)
        return dags

    def simulate(self, incoming_rate):
        results = []
        for dag in self.dags:
            total_latency = 0
            total_goodput = incoming_rate
            for node in nx.topological_sort(dag):
                node_data = dag.nodes[node]['data']
                latency, goodput = node_data.process_request(total_goodput)
                total_latency += latency
                total_goodput = goodput
            results.append({
                "total_latency": total_latency,
                "total_goodput": total_goodput
            })
        return results

    def reset(self):
        self.dags = self._generate_dags()

# Example usage:
if __name__ == "__main__":
    simulator = MicroserviceDAGSimulator(num_dags=3, max_nodes_per_dag=5)
    incoming_rate = 50  # Example incoming rate of requests

    for i in range(10):  # Simulate 10 different scenarios
        results = simulator.simulate(incoming_rate)
        print(f"Simulation {i+1} results:")
        for j, result in enumerate(results):
            print(f"  DAG {j+1}: Total Latency = {result['total_latency']:.2f}, Total Goodput = {result['total_goodput']:.2f}")
        simulator.reset()  # Reset the DAGs for a new simulation
