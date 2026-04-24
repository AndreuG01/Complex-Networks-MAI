import networkx as nx
import numpy as np
import random
import os

np.random.seed(42)
random.seed(42)

N = 1000
net_1 = nx.erdos_renyi_graph(N, 4 / (N - 1))
net_2 = nx.erdos_renyi_graph(N, 6 / (N - 1))
net_3 = nx.barabasi_albert_graph(N, 4 // 2)
net_4 = nx.barabasi_albert_graph(N, 6 // 2)

os.makedirs("networks", exist_ok=True)

graphs = {
    "er_k4": net_1,
    "er_k6": net_2,
    "ba_k4": net_3,
    "ba_k6": net_4,
}

for name, G in graphs.items():
    path = f"networks/{name}.txt"
    with open(path, "w") as f:
        f.write(f"{G.number_of_nodes()} {G.number_of_edges()}\n")
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    avg_k = 2 * G.number_of_edges() / G.number_of_nodes()
    print(f"{name}: N={G.number_of_nodes()}, E={G.number_of_edges()}, <k>={avg_k:.2f} -> {path}")

print("Done.")
