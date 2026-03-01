import networkx as nx
import os
import matplotlib.pyplot as plt


DATA_PATH = "data"

G = nx.read_pajek(os.path.join(DATA_PATH, "net5.net"))
G = nx.Graph(G)

with open(os.path.join(DATA_PATH, "positions_net5.txt"), "r") as f:
    positions = [line for line in f.read().split("\n")]

positions = {
    line.split("\t")[0]: [float(line.split("\t")[1]), float(line.split("\t")[2])] for line in positions[1:-1]
}


fig = plt.figure(figsize=(10, 5))

nx.draw_networkx_nodes(G, positions, node_size=10)
nx.draw_networkx_edges(G, positions, alpha=0.4)
# nx.draw_networkx_labels(G, positions, font_size=8)

plt.show()

