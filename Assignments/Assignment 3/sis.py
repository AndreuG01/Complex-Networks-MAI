import pandas as pd
import random
import networkx as nx
from tqdm import tqdm

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
