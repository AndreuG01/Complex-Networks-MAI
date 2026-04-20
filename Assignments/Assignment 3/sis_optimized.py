import random
from itertools import chain
import networkx as nx
from tqdm import tqdm

def SIS_simulation(
    network: nx.Graph,
    beta: float, # Probability of infection of a susceptible node by an infected neighbor (in the parctice statement, this is lambda)
    mu: float, # The recovery probability
    N_rep: int=100, # Number of repetitions of the simulation
    R0: float=0.2, # Initial fraction of infected nodes
    T_max: int=1000, # Maximum number of time steps
    T_trans: int=900 # Number of time steps of the transient period
) -> float:
    if T_trans >= T_max:
        raise ValueError("T_trans must be smaller than T_max")

    nodes = list(network.nodes())
    N = len(nodes)
    neighbor_sets = {node: set(network.neighbors(node)) for node in nodes}

    average_rho = 0.0

    for _ in tqdm(range(N_rep), desc="Simulations"):
        infected_nodes = set(random.sample(nodes, int(R0 * N)))

        avg_rho_per_timestep = 0.0
        for t in range(T_max):
            if not infected_nodes:
                break

            recoveries = {node for node in infected_nodes if random.random() < mu}

            infection_candidates = set(chain.from_iterable(
                neighbor_sets[node] for node in infected_nodes
            )) - infected_nodes

            next_infected_nodes = {
                node
                for node in infection_candidates
                if random.random() < 1 - (1 - beta) ** len(neighbor_sets[node] & infected_nodes)
            }

            infected_nodes = (infected_nodes - recoveries) | next_infected_nodes

            if t >= T_trans:
                avg_rho_per_timestep += len(infected_nodes) / N

        average_rho += avg_rho_per_timestep / (T_max - T_trans)

    return average_rho / N_rep


if __name__ == "__main__":
    N = 1000
    # Erdös-Rényi (ER) with <k>=4,
    net_1 = nx.erdos_renyi_graph(N, 4 / (N - 1))
    
    
    print(SIS_simulation(net_1, beta=0.3, mu=0.2))
    
    
    # # ER with <k>=6,
    # net_2 = nx.erdos_renyi_graph(N, 6 / (N - 1))
    # # Barábasi-Albert (BA) with <k>=4, 
    # net_3 = nx.barabasi_albert_graph(N, 4 / 2)
    # # BA with <k>=6
    # net_4 = nx.barabasi_albert_graph(N, 6 / 2)
