import pandas as pd
import random
from itertools import chain
import networkx as nx
from tqdm import tqdm


def SIS_simulation_optimized(
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



def SIS_simulation(
    network: nx.Graph,
    beta: float, # Probability of infection of a susceptible node by an infected neighbor (in the parctice statement, this is lambda)
    mu: float, # The recovery probability
    N_rep: int=100, # Number of repetitions of the simulation
    R0: int=0.2, # Initial fraction of infected nodes
    T_max: int=1000, # Maximum number of time steps
    T_trans: int=900 # Number of time steps of the transient period
) -> pd.DataFrame:
    
    N = network.number_of_nodes()
    
    average_rho = 0
    
    
    for rep in tqdm(range(N_rep), desc="Simulations"): # simulations
        num_infected = int(R0 * N)
        # print(network.nodes())
        infected_nodes = random.sample(list(network.nodes()), num_infected)
        susceptible_nodes = list(set(network.nodes()) - set(infected_nodes))
        
        t = 0
        avg_rho_per_timestep = 0
        while t < T_max and num_infected > 0: # timesteps
            next_susceptible_nodes = []
            next_infected_nodes = []
            for node in infected_nodes:
                if random.random() < mu:
                    # The current node will recover in the next timestep
                    next_susceptible_nodes.append(node)
            
            for node in susceptible_nodes:
                # Get the neigbours
                neighbors = network.neighbors(node)
                for neighbor in neighbors:
                    if neighbor in infected_nodes and random.random() < beta:
                        # The current node will be infected in the next timestep
                        next_infected_nodes.append(node)
                        break

            if t > T_trans and t < T_max:
                # Calculate the fraction of infected nodes
                avg_rho_per_timestep += len(infected_nodes) / N

            # Update the lists of infected and susceptible nodes
            infected_nodes = next_infected_nodes + [node for node in infected_nodes if node not in next_susceptible_nodes]
            num_infected = len(infected_nodes)
            susceptible_nodes = next_susceptible_nodes + [node for node in susceptible_nodes if node not in next_infected_nodes]
            
            t += 1

        average_rho += avg_rho_per_timestep / (T_max - T_trans)
        
    return average_rho / N_rep


