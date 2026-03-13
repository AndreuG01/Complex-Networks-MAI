import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


DATA_PATH = "data"
OUT_PATH = "assets"
os.makedirs(OUT_PATH, exist_ok=True)


USE_LATEX = True

if USE_LATEX:
    plt.rcParams.update({
        "text.usetex": True,
    })



def plot_degree_hist_wrong(degrees: list[int], out_path: str, savefig: bool=True, show_fig: bool=True, log_scale: bool=False):
    fig = plt.figure(figsize=(6, 6))
        
    plt.hist(degrees, color="blue", edgecolor="black", density=True) # The density parameter is used to normalize the histogram, so that the area under the histogram sums to 1, making it a probability distribution.
    
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    plt.title(f"Degree distribution{' (log scale - wrong)' if log_scale else ''}")
    
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    
    plt.tight_layout()
    if savefig:
        plt.savefig(out_path, dpi=300)
    
    if show_fig:
        plt.show()


def plot_degree_hist_logbin(degrees: list[int], out_path: str, bins: int = 10, savefig: bool=True, show_fig: bool=True):

    degrees = np.array(degrees)
    degrees = degrees + 1e-10  # Avoid log(0) problems

    k_min = degrees.min()
    k_max = degrees.max()


    log_bins = np.logspace(np.log10(k_min), np.log10(k_max), bins + 1) # http://www.mkivela.com/binning_tutorial.html
    
    
    fig = plt.figure(figsize=(6, 6))
    
    # counts, edges = np.histogram(degrees, bins=log_bins)
    # print(edges)
    # widths = edges[1:] - edges[:-1]
    # print(widths)
    # pk = counts / (len(degrees) * widths) # Normalize by the bin width is necessary for the scatter points to match the histogram's density
    # centers = np.sqrt(edges[:-1] * edges[1:])  # Geometric mean for bin centers
    # plt.plot(centers, pk, "o", color="blue")

    plt.hist(
        degrees,
        bins=log_bins,
        # density=True,
        edgecolor="black",
        color="blue",
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("$k$")
    plt.ylabel("P(k)")
    plt.title("Degree distribution (log-binned histogram)")

    if savefig:
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)

    if show_fig:
        plt.show()
        
        
def plot_degree_ccdf(degrees: list[int], out_path: str, savefig: bool=True, show_fig: bool=True):

    sorted_deg = np.array(sorted(degrees))

    k_vals = list(set(degrees))

    # CCDF: P(K >= k)
    ccdf = [(sorted_deg >= k).sum() / len(sorted_deg) for k in k_vals]

    fig = plt.figure(figsize=(6, 6))

    plt.scatter(k_vals, ccdf, color="blue", s=20)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("$k$")
    plt.ylabel("$P(K \geq k)$")
    plt.title("Degree CCDF (log-log scale)")

    if savefig:
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)

    if show_fig:
        plt.show()


def characterize_network(net_name: str, data_path: str=DATA_PATH, out_path: str=OUT_PATH):
    
    
    net_data = nx.read_pajek(os.path.join(data_path, net_name))
    G = nx.Graph(net_data)
    
    print(f"\n############ Characterization of {net_name} ############")
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    
    degrees = list(dict(G.degree()).values())
    plot_degree_hist_wrong(degrees, out_path=os.path.join(out_path, f"{net_name.replace('.', '')}_hist.png"), show_fig=False)
    plot_degree_hist_wrong(degrees, out_path=os.path.join(out_path, f"{net_name.replace('.', '')}_hist_log.png"), log_scale=True, show_fig=False)
    plot_degree_hist_logbin(
        degrees,
        out_path=os.path.join(out_path, f"{net_name.replace('.', '')}_hist_logbin.png"),
        show_fig=False
    )
    
    plot_degree_ccdf(
        degrees,
        out_path=os.path.join(out_path, f"{net_name.replace('.', '')}_ccdf.png"),
        show_fig=False
    )
    
    print(f"Max degree: {max(degrees)}")
    print(f"Min degree: {min(degrees)}")
    print(f"Average degree: {np.mean(degrees)}")
    
    clustering_coefs = list(nx.clustering(G).values())
    print(f"Average clustering coefficient: {round(np.mean(clustering_coefs), 4)}")
    
    r = nx.degree_assortativity_coefficient(G)

    print(f"Degree assortativity: {r}")
    
    print(f"Average shortest path length: {round(nx.average_shortest_path_length(G), 4)}")
    print(f"Diameter: {nx.diameter(G)}")
    
    


characterize_network("net1.net")
characterize_network("net2.net")
characterize_network("net3.net")
characterize_network("net4.net")



# G = nx.read_pajek(os.path.join(DATA_PATH, "net5.net"))
# G = nx.Graph(G)

# with open(os.path.join(DATA_PATH, "positions_net5.txt"), "r") as f:
#     positions = [line for line in f.read().split("\n")]

# positions = {
#     line.split("\t")[0]: [float(line.split("\t")[1]), float(line.split("\t")[2])] for line in positions[1:-1]
# }


# fig = plt.figure(figsize=(10, 5))

# nx.draw_networkx_nodes(G, positions, node_size=10)
# nx.draw_networkx_edges(G, positions, alpha=0.4)
# # nx.draw_networkx_labels(G, positions, font_size=8)

# plt.show()

