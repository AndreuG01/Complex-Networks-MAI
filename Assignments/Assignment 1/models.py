import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from characterization import *
from utils import *
import scipy.stats

def watts_strogatz_clustering_aspl(N: int=5000, K: int=10, num_points: int=50, true_cc: float=0.4141, true_aspl: float=5.1211):
    probs = np.logspace(-4, 0, num_points) # 10 values between 10^-4 and 10^0
    
    mean_clustering_list = []
    aspl_list = []
    for p in tqdm(probs, desc="Generating Watts-Strogatz networks and calculating metrics", total=num_points):
        net = nx.watts_strogatz_graph(N, K, p)
        clustering_coefs = list(nx.clustering(net).values())
        mean_clustering = np.mean(clustering_coefs)
        aspl = nx.average_shortest_path_length(net)
        
        mean_clustering_list.append(mean_clustering)
        aspl_list.append(aspl)
    
    fig, axes = plt.subplots(1, 1, figsize=(6, 6), sharex=True)
    axes.plot(probs, mean_clustering_list, label="Clustering Coefficient", color="blue")
    
    axes.axhline(true_cc, color="blue", linestyle="--", label=f"Clustering Coefficient of $net1$ ({true_cc})")
    
    axes.set_xscale("log")
    axes.set_xlabel("Rewiring probability ($p$)")
    axes.set_ylabel("CC", color="blue")
    axes.tick_params(axis="y", labelcolor="blue")
    

    # Add a vertical bar where the horizontal line intersects the clustering coefficient curve
    p_cc = probs[np.argmin(np.abs(np.array(mean_clustering_list) - true_cc))]
    axes.axvline(p_cc, color="gray", linestyle=":", alpha=0.8)
    
    
    
    axes2 = axes.twinx()
    axes2.plot(probs, aspl_list, label="Average Shortest Path Length", color="red")
    axes2.axhline(true_aspl, color="red", linestyle="--", label=f"ASPL of $net1$ ({true_aspl})")
    axes2.set_ylabel("$\langle l \\rangle$", color="red")
    axes2.tick_params(axis="y", labelcolor="red")
    

    # Add a vertical bar where the horizontal line intersects the ASPL curve
    p_aspl = probs[np.argmin(np.abs(np.array(aspl_list) - true_aspl))]
    axes2.axvline(p_aspl, color="gray", linestyle=":", alpha=0.8)
    
    print(f"Estimated p from clustering: {p_cc}")
    print(f"Estimated p from ASPL: {p_aspl}")
    
    axes.grid(True, which="both", alpha=0.2)
    
    plt.title("Watts-Strogatz Networks $N=5000$, $K=10$")
    plt.savefig(os.path.join(OUT_PATH, "ws_clustering_aspl.png"), dpi=300)
    

def ws_simulation(N: int=5000, K: int=10, p: float=0.15, num_trials: int=50):
    results = []
    for i in tqdm(range(num_trials), desc="Generating Watts-Strogatz networks", total=num_trials):
        net = nx.watts_strogatz_graph(N, K, p)
        res = characterize_network(net, net_name=f"ws_fake_{i}", out_path=OUT_PATH, plot=False, verbose=False)
        results.append(res)

    # Calculate the mean and std of all the metrics across the 50 runs
    metrics = results[0].keys()
    mean_metrics = {metric: np.mean([res[metric] for res in results]) for metric in metrics}
    std_metrics = {metric: np.std([res[metric] for res in results]) for metric in metrics}

    for metric in metrics:
        print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")


def compute_ccdf(network: nx.Graph):
    degrees = list(dict(network.degree()).values())
    sorted_deg = np.array(sorted(degrees))
    k_vals = list(set(degrees))
    
    # CCDF: P(K >= k)
    ccdf = [(sorted_deg >= k).sum() / len(sorted_deg) for k in k_vals]
    
    return k_vals, ccdf


def fit_power_law(degrees: list[int], ccdf: list[float]=None):
    sorted_deg = np.array(sorted(degrees))
    k_vals = list(set(degrees))
    
    # CCDF: P(K >= k)
    if ccdf is None:
        ccdf = [(sorted_deg >= k).sum() / len(sorted_deg) for k in k_vals]

    m, b = np.polyfit(np.log(k_vals), np.log(ccdf), 1)
    gamma = 1 - m
    return gamma, m, b


def ba_simulation(N: int=5000, m: int=5, num_trials: int=50):
    results = []
    for i in tqdm(range(num_trials), desc="Generating Barabási-Albert networks", total=num_trials):
        net = nx.barabasi_albert_graph(N, m)
        res = characterize_network(net, net_name=f"ba_fake_{i}", out_path=OUT_PATH, plot=False, verbose=False)
        results.append(res)

    # Calculate the mean and std of all the metrics across the 50 runs
    metrics = results[0].keys()
    mean_metrics = {metric: np.mean([res[metric] for res in results]) for metric in metrics}
    std_metrics = {metric: np.std([res[metric] for res in results]) for metric in metrics}

    for metric in metrics:
        print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")


def _run_er_trial(i: int, N: int, p: float):
    net = nx.gnm_random_graph(N, 24873)
    # net = nx.erdos_renyi_graph(N, p)
    was_connected = nx.is_connected(net)
    if not was_connected:
        # If the graph is not connected, keep only the largest connected component.
        net = net.subgraph(max(nx.connected_components(net), key=len)).copy()

    res = characterize_network(net, net_name=f"er_fake_{i}", out_path=OUT_PATH, plot=False, verbose=False)
    return i, res, was_connected


def er_simulation(N: int=5000, p: float=0.0021, num_trials: int=50, max_workers: int | None=None):
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_er_trial, i, N, p) for i in range(num_trials)]

        disconnected_count = 0
        for future in tqdm(as_completed(futures), desc="Generating Erdös-Rényi networks", total=num_trials):
            _, res, was_connected = future.result()
            if not was_connected:
                disconnected_count += 1
            results.append(res)

    if disconnected_count > 0:
        print(f"{disconnected_count}/{num_trials} generated ER networks were not connected.")

    # Calculate the mean and std of all the metrics across the 50 runs
    metrics = results[0].keys()
    mean_metrics = {metric: np.mean([res[metric] for res in results]) for metric in metrics}
    std_metrics = {metric: np.std([res[metric] for res in results]) for metric in metrics}

    for metric in metrics:
        print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")



def cm_simulation(degree_sequence: list[int], num_trials: int=50):
    results = []
    degrees_cm = []
    for i in tqdm(range(num_trials), desc="Generating Configuration Model networks", total=num_trials):
        net = nx.configuration_model(degree_sequence, create_using=None)
        net = nx.Graph(net)  # Remove parallel edges and self-loops
        degree_sequence_cm = list(dict(net.degree()).values())
        degrees_cm.append(degree_sequence_cm)
        res = characterize_network(net, net_name=f"cm_fake_{i}", out_path=OUT_PATH, plot=False, verbose=False)
        results.append(res)

    # Calculate the mean and std of all the metrics across the 50 runs
    metrics = results[0].keys()
    mean_metrics = {metric: np.mean([res[metric] for res in results]) for metric in metrics}
    std_metrics = {metric: np.std([res[metric] for res in results]) for metric in metrics}
    
    # Plot the spearman correlation between the degree sequence of the original network and the degree sequence of the configuration model networks
    average_degrees_cm = np.mean(degrees_cm, axis=0)
    fig = plt.figure(figsize=(6, 6))
    plt.plot([min(degree_sequence), max(degree_sequence)], [min(degree_sequence), max(degree_sequence)], color="black", label="$y=x$")
    plt.scatter(degree_sequence, average_degrees_cm, color="blue", s=20)
    plt.xlabel("Original degree sequence")
    plt.ylabel("Configuration model degree sequence")
    
    spearman_corr, _ = scipy.stats.spearmanr(degree_sequence, average_degrees_cm)
    plt.title(f"Degree sequence correlation between original and CM networks\n($\\rho_{{Spearman}}$ = {spearman_corr:.4f})")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, "cm_degree_correlation.png"), dpi=300)
    plt.show()

    for metric in metrics:
        print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")




def random_gemoetric_graph(positions: dict[str, list[float]], radius: float) -> nx.Graph:
    G = nx.Graph()
    for node, pos in positions.items():
        G.add_node(node, pos=pos)

    nodes = list(G.nodes(data=True))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_i, data_i = nodes[i]
            node_j, data_j = nodes[j]
            pos_i = np.array(data_i["pos"])
            pos_j = np.array(data_j["pos"])
            distance = np.linalg.norm(pos_i - pos_j)
            if distance <= radius:
                G.add_edge(node_i, node_j)

    return G


def random_geometric_graph_simulation(positions: dict[str, list[float]], radius: float, show_fig: bool=False):
    random_gemoetric_graph_net5 = random_gemoetric_graph(positions, radius=radius)
    components = list(nx.connected_components(random_gemoetric_graph_net5))
    print(f"Number of connected components in the random geometric graph: {len(components)}")
    fig = plt.figure(figsize=(6, 6))
    plt.title(f"Random geometric graph with radius {radius}")

    for i, component in enumerate(components):
        nx.draw_networkx_nodes(
            random_gemoetric_graph_net5,
            positions,
            nodelist=list(component),
            node_size=20,
            node_color=colors[i % len(colors)],
        )
    nx.draw_networkx_edges(random_gemoetric_graph_net5, positions, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, f"net5_random_geometric_graph_radius_{radius}.png"), dpi=300)
    if show_fig:
        plt.show()



def power_law(net_name: str):
    net = load_network(net_name)

    k_vals, ccdf = compute_ccdf(net)
    fit_gamma, fit_m, fit_b = fit_power_law(k_vals, ccdf)
    print(f"Fitted power-law exponent (gamma): {fit_gamma:.4f}")

    fig = plt.figure(figsize=(6, 6))
    plt.plot(np.log(k_vals), np.log(ccdf), "o", color="green", label="Data")
    plt.plot(np.log(k_vals), fit_m * np.log(k_vals) + fit_b, label=f"Power-law fit ($\gamma= {fit_gamma:.4f}$)", color="black")
    theoretical_gamma = 3
    theoretical_m = 1 - theoretical_gamma
    theoretical_b = np.log(ccdf[0]) - theoretical_m * np.log(k_vals[0])  # Ensure the theoretical line goes through the first point of the CCDF
    plt.plot(np.log(k_vals), theoretical_m * np.log(k_vals) + theoretical_b, label=f"Power-law $\gamma= {theoretical_gamma}$", color="orange", linestyle="--")
    plt.legend()
    plt.xlabel("$\log(k)$")
    plt.ylabel("$\log(P(K \geq k))$")
    plt.title(f"Degree CCDF of {net_name} with Power-law Fit")
    plt.tight_layout()
    plt.grid(True, which="both", alpha=0.2)
    plt.savefig(os.path.join(OUT_PATH, f"{net_name}_powerlaw_ccdf_fit.png"), dpi=300)
    plt.show()
  
