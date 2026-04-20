import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import networkx as nx
import pandas as pd
from sis import *

ASSETS_PATH = "assets/"


def run_simulation(network, beta, mu, optimized):
    if optimized:
        return SIS_simulation_optimized(network, beta, mu)
    else:
        return SIS_simulation(network, beta, mu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--efficient",
        action="store_true",
        help="Tag stored results as efficient",
    )
    args = parser.parse_args()
    
    np.random.seed(42)
    random.seed(42)
    
    N = 1000
    # Erdös-Rényi (ER) with <k>=4,
    net_1 = nx.erdos_renyi_graph(N, 4 / (N - 1))
    # ER with <k>=6,
    net_2 = nx.erdos_renyi_graph(N, 6 / (N - 1))
    # Barábasi-Albert (BA) with <k>=4, 
    net_3 = nx.barabasi_albert_graph(N, 4 // 2)
    # BA with <k>=6
    net_4 = nx.barabasi_albert_graph(N, 6 // 2)

    
    betas = np.arange(0, 0.31, 0.01)
    os.makedirs("results", exist_ok=True)

    graphs = {
        "er_k4": net_1,
        "er_k6": net_2,
        "ba_k4": net_3,
        "ba_k6": net_4,
    }

    for graph_name, graph in graphs.items():
        results_mu1_rows = []
        for beta in betas:
            start_time = time.perf_counter()
            rho = run_simulation(graph, beta, mu=0.2, optimized=args.efficient)
            elapsed = time.perf_counter() - start_time
            results_mu1_rows.append({
                "beta": beta,
                "rho": rho,
                "time_seconds": elapsed,
            })

        results_mu1 = pd.DataFrame(results_mu1_rows)
        results_mu1.to_csv(f"results/{graph_name}_mu_0.2{'_efficient' if args.efficient else ''}.csv", index=False)
        
        results_mu2_rows = []
        for beta in betas:
            start_time = time.perf_counter()
            rho = run_simulation(graph, beta, mu=0.4, optimized=args.efficient)
            elapsed = time.perf_counter() - start_time
            results_mu2_rows.append({
                "beta": beta,
                "rho": rho,
                "time_seconds": elapsed,
            })

        results_mu2 = pd.DataFrame(results_mu2_rows)
        results_mu2.to_csv(f"results/{graph_name}_mu_0.4{'_efficient' if args.efficient else ''}.csv", index=False)
