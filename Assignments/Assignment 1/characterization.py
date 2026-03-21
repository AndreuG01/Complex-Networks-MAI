import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
from utils import OUT_PATH


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



def characterize_network(G: nx.Graph, net_name: str, out_path: str=OUT_PATH, plot: bool=True, verbose: bool=True) -> dict:
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    
    degrees = list(dict(G.degree()).values())
    if plot:
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
    
    max_degree = max(degrees)
    min_degree = min(degrees)
    avg_degree = np.mean(degrees)


    clustering_coefs = list(nx.clustering(G).values())
    avg_clustering = np.mean(clustering_coefs)

    r = nx.degree_assortativity_coefficient(G)
    avg_spl = nx.average_shortest_path_length(G)
        
    diameter = nx.diameter(G)
    
    if verbose:
        print(f"\n############ Characterization of {net_name} ############")
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")
        print(f"Max degree: {max_degree}")
        print(f"Min degree: {min_degree}")
        print(f"Average degree: {avg_degree}")
        print(f"Average clustering coefficient: {round(avg_clustering, 4)}")
        print(f"Degree assortativity: {r}")
        print(f"Average shortest path length: {round(avg_spl, 4)}")
        print(f"Diameter: {diameter}")
    
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "max_degree": max_degree,
        "min_degree": min_degree,
        "avg_degree": avg_degree,
        "avg_clustering": avg_clustering,
        "assortativity": r,
        "aspl": avg_spl,
        "diameter": diameter,
    }


def top_n(centrality_dict, n=5):
    return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]


def microscopic_description(G: nx.Graph, net_name: str, verbose: bool=True) -> dict:
    

    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G)

    # Eigenvector centrality can fail on disconnected graphs
    try:
        eigen_cent = nx.eigenvector_centrality_numpy(G)
    except Exception as e:
        print(f"Eigenvector centrality computation failed: {e}")
        eigen_cent = {}


    top_degree = top_n(degree_cent, 5)
    top_betweenness = top_n(between_cent, 5)
    top_eigen = top_n(eigen_cent, 5) if eigen_cent else []

    if verbose:
        print(f"\n############ Microscopic description of {net_name} ############")
        print("\nTop 5 nodes by Degree Centrality:")
        for node, score in top_degree:
            print(f"  {node}: {score:.4f}")

        print("\nTop 5 nodes by Betweenness Centrality:")
        for node, score in top_betweenness:
            print(f"  {node}: {score:.4f}")

        print("\nTop 5 nodes by Eigenvector Centrality:")
        if top_eigen:
            for node, score in top_eigen:
                print(f"  {node}: {score:.4f}")
        else:
            print("  eigenvector centrality not available for this graph.")

    return {
        "top_degree": top_degree,
        "top_betweenness": top_betweenness,
        "top_eigen": top_eigen,
    }