import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tqdm import tqdm


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


def load_network(net_name: str, data_path: str=DATA_PATH) -> nx.Graph:
    net_data = nx.read_pajek(os.path.join(data_path, net_name))
    G = nx.Graph(net_data)
    return G


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


def microscopic_description(G: nx.Graph, net_name: str="network") -> dict:
    """Prints and returns top 5 nodes by degree, betweenness, and eigenvector centrality."""
    print(f"\n############ Microscopic description of {net_name} ############")

    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G)

    # Eigenvector centrality can fail on disconnected graphs; use numpy fallback and catch exceptions
    try:
        eigen_cent = nx.eigenvector_centrality_numpy(G)
    except Exception as e:
        print(f"Eigenvector centrality computation failed: {e}")
        eigen_cent = {}

    def top_n(centrality_dict, n=5):
        return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]

    top_degree = top_n(degree_cent, 5)
    top_betweenness = top_n(between_cent, 5)
    top_eigen = top_n(eigen_cent, 5) if eigen_cent else []

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

# characterize_network(load_network("net1.net"))
# characterize_network(load_network("net2.net"))
# characterize_network(load_network("net3.net"))
# characterize_network(load_network("net4.net"))

microscopic_description(load_network("net1.net"), 'net1.net')
microscopic_description(load_network("net2.net"), 'net2.net')
microscopic_description(load_network("net3.net"), 'net3.net')
microscopic_description(load_network("net4.net"),  'net4.net')


# def watts_strogatz_clustering_aspl(N: int=5000, K: int=10, num_points: int=50, true_cc: float=0.4141, true_aspl: float=5.1211):
#     probs = np.logspace(-4, 0, num_points) # 10 values between 10^-4 and 10^0
    
#     mean_clustering_list = []
#     aspl_list = []
#     for p in tqdm(probs, desc="Generating Watts-Strogatz networks and calculating metrics", total=num_points):
#         net = nx.watts_strogatz_graph(N, K, p)
#         clustering_coefs = list(nx.clustering(net).values())
#         mean_clustering = np.mean(clustering_coefs)
#         aspl = nx.average_shortest_path_length(net)
        
#         mean_clustering_list.append(mean_clustering)
#         aspl_list.append(aspl)
    
#     fig, axes = plt.subplots(1, 1, figsize=(6, 6), sharex=True)
#     axes.plot(probs, mean_clustering_list, label="Clustering Coefficient", color="blue")
    
#     axes.axhline(true_cc, color="blue", linestyle="--", label=f"Clustering Coefficient of $net1$ ({true_cc})")
    
#     axes.set_xscale("log")
#     axes.set_xlabel("Rewiring probability ($p$)")
#     axes.set_ylabel("CC", color="blue")
#     axes.tick_params(axis="y", labelcolor="blue")
    

#     # Add a vertical bar where the horizontal line intersects the clustering coefficient curve
#     p_cc = probs[np.argmin(np.abs(np.array(mean_clustering_list) - true_cc))]
#     axes.axvline(p_cc, color="gray", linestyle=":", alpha=0.8)
    
    
    
#     axes2 = axes.twinx()
#     axes2.plot(probs, aspl_list, label="Average Shortest Path Length", color="red")
#     axes2.axhline(true_aspl, color="red", linestyle="--", label=f"ASPL of $net1$ ({true_aspl})")
#     axes2.set_ylabel("$\langle l \\rangle$", color="red")
#     axes2.tick_params(axis="y", labelcolor="red")
    

#     # Add a vertical bar where the horizontal line intersects the ASPL curve
#     p_aspl = probs[np.argmin(np.abs(np.array(aspl_list) - true_aspl))]
#     axes2.axvline(p_aspl, color="gray", linestyle=":", alpha=0.8)
    
#     print(f"Estimated p from clustering: {p_cc}")
#     print(f"Estimated p from ASPL: {p_aspl}")
    
#     axes.grid(True, which="both", alpha=0.2)
    
#     plt.title("Watts-Strogatz Networks $N=5000$, $K=10$")
#     plt.savefig(os.path.join(OUT_PATH, "ws_clustering_aspl.png"), dpi=300)


# def ws_simulation(N: int=5000, K: int=10, p: float=0.15, num_trials: int=50):
#     results = []
#     for i in tqdm(range(num_trials), desc="Generating Watts-Strogatz networks", total=num_trials):
#         net = nx.watts_strogatz_graph(N, K, p)
#         res = characterize_network(net, net_name=f"ws_fake_{i}", out_path=OUT_PATH, plot=False, verbose=False)
#         results.append(res)

#     # Calculate the mean and std of all the metrics across the 50 runs
#     metrics = results[0].keys()
#     mean_metrics = {metric: np.mean([res[metric] for res in results]) for metric in metrics}
#     std_metrics = {metric: np.std([res[metric] for res in results]) for metric in metrics}

#     for metric in metrics:
#         print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")


# def compute_ccdf(network: nx.Graph):
#     degrees = list(dict(network.degree()).values())
#     sorted_deg = np.array(sorted(degrees))
#     k_vals = list(set(degrees))
    
#     # CCDF: P(K >= k)
#     ccdf = [(sorted_deg >= k).sum() / len(sorted_deg) for k in k_vals]
    
#     return k_vals, ccdf


# def fit_power_law(degrees: list[int], ccdf: list[float]=None):
#     sorted_deg = np.array(sorted(degrees))
#     k_vals = list(set(degrees))
    
#     # CCDF: P(K >= k)
#     if ccdf is None:
#         ccdf = [(sorted_deg >= k).sum() / len(sorted_deg) for k in k_vals]

#     m, b = np.polyfit(np.log(k_vals), np.log(ccdf), 1)
#     gamma = 1 - m
#     return gamma, m, b



# # clustering_coefs_list = []
# # aspl = []
# # probs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# # for p in probs:
# #     print(p)
# #     net1_fake = nx.watts_strogatz_graph(5000, 10, p)
# #     clustering_coefs = list(nx.clustering(net1_fake).values())
# #     mean_clustering = np.mean(clustering_coefs)
# #     mean_spl = nx.average_shortest_path_length(net1_fake)
    
# #     aspl.append(mean_spl)
# #     clustering_coefs_list.append(mean_clustering)

# # fig = plt.figure(figsize=(6, 6))

# # plt.plot(probs, clustering_coefs_list, "o-", label="Clustering Coefficient")
# # plt.plot(probs, aspl, "s-", label="Average Shortest Path Length")

# # plt.xscale("log")

# # plt.savefig("ws_clustering_aspl.png", dpi=300)

# # net1_fake = nx.watts_strogatz_graph(5000, 10, 0.1)
# # characterize_network(net1_fake, net_name='net1_fake', out_path=OUT_PATH)

# # net2_fake = nx.erdos_renyi_graph(5000, 0.0021)
# # characterize_network(net2_fake, net_name='net2_fake', out_path=OUT_PATH)


# def power_law(net_name: str):
#     net = load_network(net_name)

#     k_vals, ccdf = compute_ccdf(net)
#     fit_gamma, fit_m, fit_b = fit_power_law(k_vals, ccdf)
#     print(f"Fitted power-law exponent (gamma): {fit_gamma:.4f}")

#     fig = plt.figure(figsize=(6, 6))
#     plt.plot(np.log(k_vals), np.log(ccdf), "o", color="green", label="Data")
#     plt.plot(np.log(k_vals), fit_m * np.log(k_vals) + fit_b, label=f"Power-law fit ($\gamma= {fit_gamma:.4f}$)", color="black")
#     theoretical_gamma = 3
#     theoretical_m = 1 - theoretical_gamma
#     theoretical_b = np.log(ccdf[0]) - theoretical_m * np.log(k_vals[0])  # Ensure the theoretical line goes through the first point of the CCDF
#     plt.plot(np.log(k_vals), theoretical_m * np.log(k_vals) + theoretical_b, label=f"Power-law $\gamma= {theoretical_gamma}$", color="orange", linestyle="--")
#     plt.legend()
#     plt.xlabel("$\log(k)$")
#     plt.ylabel("$\log(P(K \geq k))$")
#     plt.title(f"Degree CCDF of {net_name} with Power-law Fit")
#     plt.tight_layout()
#     plt.grid(True, which="both", alpha=0.2)
#     plt.savefig(os.path.join(OUT_PATH, f"{net_name}_powerlaw_ccdf_fit.png"), dpi=300)


# def ba_simulation(N: int=5000, m: int=5, num_trials: int=50):
#     results = []
#     for i in tqdm(range(num_trials), desc="Generating Barabási-Albert networks", total=num_trials):
#         net = nx.barabasi_albert_graph(N, m)
#         res = characterize_network(net, net_name=f"ba_fake_{i}", out_path=OUT_PATH, plot=False, verbose=False)
#         results.append(res)

#     # Calculate the mean and std of all the metrics across the 50 runs
#     metrics = results[0].keys()
#     mean_metrics = {metric: np.mean([res[metric] for res in results]) for metric in metrics}
#     std_metrics = {metric: np.std([res[metric] for res in results]) for metric in metrics}

#     for metric in metrics:
#         print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")


# # ba_simulation()


# def cm_simulation(degree_sequence: list[int], num_trials: int=50):
#     results = []
#     degrees_cm = []
#     for i in tqdm(range(num_trials), desc="Generating Configuration Model networks", total=num_trials):
#         net = nx.configuration_model(degree_sequence, create_using=None)
#         net = nx.Graph(net)  # Remove parallel edges and self-loops
#         degree_sequence_cm = list(dict(net.degree()).values())
#         degrees_cm.append(degree_sequence_cm)
#         res = characterize_network(net, net_name=f"cm_fake_{i}", out_path=OUT_PATH, plot=False, verbose=False)
#         results.append(res)

#     # Calculate the mean and std of all the metrics across the 50 runs
#     metrics = results[0].keys()
#     mean_metrics = {metric: np.mean([res[metric] for res in results]) for metric in metrics}
#     std_metrics = {metric: np.std([res[metric] for res in results]) for metric in metrics}
    
#     # Plot the spearman correlation between the degree sequence of the original network and the degree sequence of the configuration model networks
#     average_degrees_cm = np.mean(degrees_cm, axis=0)
#     fig = plt.figure(figsize=(6, 6))
#     plt.plot([min(degree_sequence), max(degree_sequence)], [min(degree_sequence), max(degree_sequence)], color="black", label="$y=x$")
#     plt.scatter(degree_sequence, average_degrees_cm, color="blue", s=20)
#     plt.xlabel("Original degree sequence")
#     plt.ylabel("Configuration model degree sequence")
#     import scipy.stats
#     spearman_corr, _ = scipy.stats.spearmanr(degree_sequence, average_degrees_cm)
#     plt.title(f"Degree sequence correlation between original and CM networks\n($\\rho_{{Spearman}}$ = {spearman_corr:.4f})")
#     plt.grid(True, alpha=0.2)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUT_PATH, "cm_degree_correlation.png"), dpi=300)

#     for metric in metrics:
#         print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

# # net_3 = load_network("net3.net")
# # cm_simulation(list(dict(net_3.degree()).values()), num_trials=50)


# # power_law("net3.net")
# # power_law("net4.net")

# # net3 = load_network("net3.net")

# # net3_fake = nx.configuration_model([degree for _, degree in net3.degree()], create_using=None)
# # characterize_network(net3_fake, net_name='net3_fake', out_path=OUT_PATH)
# # degrees = list(dict(net3_fake.degree()).values())
# # sorted_deg = np.array(sorted(degrees))
# # k_vals = list(set(degrees))
# # # CCDF: P(K >= k)
# # ccdf = [(sorted_deg >= k).sum() / len(sorted_deg) for k in k_vals]

# # fig = plt.plot(np.log(k_vals), np.log(ccdf), "o", color="blue")
# # # plt.xscale("log")
# # # plt.yscale("log")

# # m, b = np.polyfit(np.log(k_vals), np.log(ccdf), 1)
# # # ln(y) = m*ln(x) + b -> y = exp(m*ln(x) + b) = x^m * exp(b)
# # # m =  - (gamma - 1)

# # gamma = 1 - m
# # y = k_vals ** m * np.exp(b)
# # plt.plot(np.log(k_vals), m * np.log(k_vals) + b, label=f"Gamma: {gamma}", color="black")
# # plt.legend()
# # plt.savefig("net3_fake_ccdf.png", dpi=300)


# # Visualize the network

# net5 = nx.read_pajek(os.path.join(DATA_PATH, "net5.net"))
# net5 = nx.Graph(net5)

# print(f"Number of nodes in net5: {net5.number_of_nodes()}")
# print(f"Number of edges in net5: {net5.number_of_edges()}")

# with open(os.path.join(DATA_PATH, "positions_net5.txt"), "r") as f:
#     positions = [line for line in f.read().split("\n")]

# positions = {
#     line.split("\t")[0]: [float(line.split("\t")[1]), float(line.split("\t")[2])] for line in positions[1:-1]
# }


# fig = plt.figure(figsize=(6, 6))
# plt.title("Visualization of net5")
# net_5_components = list(nx.connected_components(net5))

# colors = [
#     "#640D5F",
#     "#D91656",
#     "#EB5B00",
#     "#FFB200",
#     "#2D728F",
#     "#3E8914",
#     "#FF00D3",
# ]
# for i, component in enumerate(net_5_components):
#     nx.draw_networkx_nodes(
#         net5,
#         positions,
#         nodelist=list(component),
#         node_size=20,
#         node_color=colors[i],
#     )
# nx.draw_networkx_edges(net5, positions, alpha=0.4)
# # nx.draw_networkx_labels(G, positions, font_size=8)
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_PATH, "net5_visualization.png"), dpi=300)


# # Is the network connected?
# # No
# print(f"Network connected: {nx.is_connected(net5)}")


# # Is it scale-free (follows a power-law degree distribution)?
# net_name = "net5"
# degrees_net5 = list(dict(net5.degree()).values())

# plot_degree_hist_wrong(degrees_net5, out_path=os.path.join(OUT_PATH, f"{net_name.replace('.', '')}_hist.png"), show_fig=False)
# plot_degree_hist_wrong(degrees_net5, out_path=os.path.join(OUT_PATH, f"{net_name.replace('.', '')}_hist_log.png"), log_scale=True, show_fig=False)
# plot_degree_hist_logbin(
#     degrees_net5,
#     out_path=os.path.join(OUT_PATH, f"{net_name.replace('.', '')}_hist_logbin.png"),
#     show_fig=False
# )

# plot_degree_ccdf(
#     degrees_net5,
#     out_path=os.path.join(OUT_PATH, f"{net_name.replace('.', '')}_ccdf.png"),
#     show_fig=False
# )
# # NOT A POWER LAW 

# # Is the largest connected component a small-world network (high clustering coefficent and low aspl)?
# print(f"Number of connected components: {len(net_5_components)}")
# print(f"Size connected components: {[len(c) for c in net_5_components]}")
# max_cc = max(net_5_components, key=len)
# max_cc_graph = net5.subgraph(max_cc)

# # Plot the largest connected component
# fig = plt.figure(figsize=(6, 6))
# plt.title("Visualization of the largest connected component of net5")
# nx.draw_networkx_nodes(
#     max_cc_graph,
#     positions,
#     node_size=20,
#     node_color=colors[0],
# )
# nx.draw_networkx_edges(max_cc_graph, positions, alpha=0.4)
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_PATH, "net5_largest_cc_visualization.png"), dpi=300)


# print(f"Number of nodes in the largest CC: {max_cc_graph.number_of_nodes()}")
# print(f"Number of edges in the largest CC: {max_cc_graph.number_of_edges()}")
# degree_list = list(dict(max_cc_graph.degree()).values())
# print(f"Max degree in the largest CC: {max(degree_list)}")
# print(f"Min degree in the largest CC: {min(degree_list)}")
# print(f"Average degree in the largest CC: {round(np.mean(degree_list), 4)}")

# print(f"Clustering coefficient of the largetst CC: {round(np.mean(list(nx.clustering(max_cc_graph).values())), 4)}")
# print(f"Degree assortativity of the largest CC: {round(nx.degree_assortativity_coefficient(max_cc_graph), 4)}")
# print(f"Average shortest path length of the largest CC: {round(nx.average_shortest_path_length(max_cc_graph), 4)}")
# print(f"Diameter of the largest CC: {nx.diameter(max_cc_graph)}")
# # Propose algorithm

# # Random geometric graph


# def random_gemoetric_graph(positions: dict[str, list[float]], radius: float) -> nx.Graph:
#     G = nx.Graph()
#     for node, pos in positions.items():
#         G.add_node(node, pos=pos)

#     nodes = list(G.nodes(data=True))
#     for i in range(len(nodes)):
#         for j in range(i + 1, len(nodes)):
#             node_i, data_i = nodes[i]
#             node_j, data_j = nodes[j]
#             pos_i = np.array(data_i["pos"])
#             pos_j = np.array(data_j["pos"])
#             distance = np.linalg.norm(pos_i - pos_j)
#             if distance <= radius:
#                 G.add_edge(node_i, node_j)

#     return G


# distances = []
# for u, v in net5.edges():
#     pos_u = np.array(positions[u])
#     pos_v = np.array(positions[v])
#     distance = np.linalg.norm(pos_u - pos_v) # Assume euclidean distance.
#     distances.append(distance)

# print(f"r: {max(distances):.4f}")

# plt.close("all")
# for radius in [0.001, 0.0005, 0.05, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25]:
#     random_gemoetric_graph_net5 = random_gemoetric_graph(positions, radius=radius)
#     components = list(nx.connected_components(random_gemoetric_graph_net5))
#     print(f"Number of connected components in the random geometric graph: {len(components)}")
#     fig = plt.figure(figsize=(6, 6))
#     plt.title(f"Random geometric graph with radius {radius}")

#     for i, component in enumerate(components):
#         nx.draw_networkx_nodes(
#             random_gemoetric_graph_net5,
#             positions,
#             nodelist=list(component),
#             node_size=20,
#             node_color=colors[i % len(colors)],
#         )
#     nx.draw_networkx_edges(random_gemoetric_graph_net5, positions, alpha=0.4)

#     plt.tight_layout()
#     plt.savefig(os.path.join(OUT_PATH, f"net5_random_geometric_graph_radius_{radius}.png"), dpi=300)