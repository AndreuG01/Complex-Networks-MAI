import pandas as pd
import numpy as np
from tqdm.asyncio import tqdm
from utils import extract_filename_info, generate_colormap, match_communities, reorder_communities
import matplotlib.pyplot as plt
import networkx as nx
import os
import math
import infomap as im
from cdlib import NodeClustering
from cdlib.evaluation import normalized_mutual_information, variation_of_information




def get_synthetic_ground_truth_partition(nodes, n_blocks: int):
    
    nodes_int = sorted(int(node) for node in nodes)
    n_nodes = len(nodes_int)
    if n_nodes == 0:
        return []

    block_size = n_nodes // n_blocks

    communities = [[] for _ in range(n_blocks)]
    for node in nodes_int:
        block_idx = min((node - 1) // block_size, n_blocks - 1)
        communities[block_idx].append(str(node))

    return communities


def partition_to_labels(nodes, communities):
    node_order = sorted(nodes, key=lambda x: int(x))
    node_to_comm = {}

    for comm_idx, comm in enumerate(communities):
        for node in comm:
            node_key = str(node)
            node_to_comm[node_key] = comm_idx

    labels = np.array([node_to_comm[str(node)] for node in node_order], dtype=np.int64)
    return labels


def partition_jaccard_index(labels_true, labels_pred):
    n = len(labels_pred)

    intersection = 0
    union = 0

    for i in range(n):
        for j in range(i + 1, n):
            same_pred = labels_pred[i] == labels_pred[j]
            same_true = labels_true[i] == labels_true[j]

            if same_pred or same_true:
                union += 1
                if same_pred and same_true:
                    intersection += 1

    if union == 0:
        return 1.0

    return intersection / union


def communities_to_node_clustering(communities):
    # As the community detection is not done using the cdlib library, the communities need to be converted into the format
    # so that the evaluation metrics from cdlib can be used.
    return NodeClustering(
        communities=[[str(node) for node in comm] for comm in communities],
        graph=None,
        method_name="partition",
    )


def partition_normalized_mutual_information(true_partition, predicted_partition):
    result = normalized_mutual_information(
        communities_to_node_clustering(true_partition),
        communities_to_node_clustering(predicted_partition)
    )
    return float(result.score)


def partition_normalized_variation_of_information(true_partition, predicted_partition, n_nodes: int):
    
    result = variation_of_information(
        communities_to_node_clustering(true_partition),
        communities_to_node_clustering(predicted_partition)
    )
    nvi = float(result.score) / math.log(n_nodes)
    return float(np.clip(nvi, 0.0, 1.0))



def plot_degree_distributions(G, filepath=None, prr=None, out_path=None, savefig=True, show_fig=False):
    degree_sequence = [degree for _, degree in G.degree()]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bins = np.arange(min(degree_sequence), max(degree_sequence) + 2) - 0.5
    ax.hist(degree_sequence, bins=bins, edgecolor='black', color="blue")

    ax.set_xlabel("Degree")
    ax.set_ylabel("Nodes")

    if prr is not None:
        ax.set_title(f'prr = {prr}')

    fig.tight_layout()

    if savefig:
        
        plot_name = f'prr_{prr}_degree_distribution.png'
        plt.savefig(os.path.join(out_path, plot_name), dpi=300, bbox_inches='tight')

    if show_fig:
        plt.show()

    plt.close(fig)

def visualize_network(
    G,
    pos,
    out_path: str,
    savefig: bool = True,
    show_fig: bool = False,
    prr=None,
    communities: list[list[int]] = None,
    max_communities: int = 20
):
    fig, ax = plt.subplots(figsize=(5, 5))

    if communities is None:
        nx.draw(
            G, pos, ax=ax,
            with_labels=False,
            node_color="blue",
            edge_color="gray",
            node_size=20,
            width=0.1
        )
    else:
        community_index = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_index[node] = i

        color_nodes = [community_index[node] for node in G.nodes()]

        cmap = generate_colormap()
        norm = plt.Normalize(vmin=0, vmax=max_communities - 1)

        nx.draw(
            G, pos, ax=ax,
            with_labels=False,
            node_color=color_nodes,
            cmap=cmap,
            edge_color="lightgrey",
            node_size=20,
            width=0.1,
            vmin=0,
            vmax=max_communities - 1
        )


    if communities is not None:
        ax.set_title(f"prr = {prr}. {len(communities)} {'communities' if len(communities) != 1 else 'community'} detected.")
    else:
        ax.set_title(f"prr = {prr}")

    if savefig:
        plt.savefig(os.path.join(out_path, f"prr_{prr}.png"), dpi=300, bbox_inches="tight")

    if show_fig:
        plt.show()

    plt.close(fig)



def detect_communities(G, algorithm_name='louvain'):    
    communities = []
    if algorithm_name.lower() == 'greedy':
        communities = [list(com) for com in nx.community.greedy_modularity_communities(G)]
        
    elif algorithm_name.lower() == 'louvain':
        modularity_louvain = -1
        best_communities_louvain = None
        for realization in range(10):
            comms = nx.community.louvain_communities(G, seed=realization)
            mod = nx.community.modularity(G, comms)
            if mod > modularity_louvain:
                modularity_louvain = mod
                best_communities_louvain = comms
        communities = [list(com) for com in best_communities_louvain]
        
    elif algorithm_name.lower() == 'infomap':
        # Infomap expects integer node IDs, so we map string IDs to integers and back
        mapping = {node: i for i, node in enumerate(G.nodes())}
        reverse_mapping = {i: node for node, i in mapping.items()}
        
        im_model = im.Infomap(silent=True)
        for u, v in G.edges():
            im_model.add_link(mapping[u], mapping[v])
        im_model.run()
        
        # Extract communities into a list of sets
        comm_dict = {}
        for node in im_model.tree:
            if node.is_leaf:
                module = node.module_id
                nx_node = reverse_mapping[node.node_id]
                if module not in comm_dict:
                    comm_dict[module] = set()
                comm_dict[module].add(nx_node)
        communities = [list(com) for com in comm_dict.values()]
    
    # Ensure every node is assigned.
    # If there are some nodes that are isolated, some algorithms might not assign them to any community.
    # We will put them in their own singleton communities.
    assigned_nodes = [node for comm in communities for node in comm]
    missing_nodes = [node for node in G.nodes() if node not in set(assigned_nodes)]
    if missing_nodes:
        communities.extend([[node] for node in missing_nodes])
        assigned_nodes = [node for comm in communities for node in comm]


    return communities

    
def community_metrics(G, communities):
    num_communities = len(communities)
    modularity = nx.community.modularity(G, communities)
    return num_communities, modularity

def plot_metrics_evolution(metrics, show_fig=True, savefig=False, out_path=None, metric_name: str="Number of communities"):
    ordered_prr = sorted(metrics.keys())
    fig = plt.figure(figsize=(7, 5))
    plt.title(f"{metric_name} vs prr")
    
    custom_cmap = generate_colormap()
    plt.plot(ordered_prr, [metrics[k]["greedy"] for k in ordered_prr], label='Greedy', color=custom_cmap(0))
    plt.plot(ordered_prr, [metrics[k]["louvain"] for k in ordered_prr], label='Louvain', color=custom_cmap(1), linestyle="--")
    plt.plot(ordered_prr, [metrics[k]["infomap"] for k in ordered_prr], label='Infomap', color=custom_cmap(2), linestyle=":")
    plt.xlabel("prr")
    plt.grid(alpha=0.2)
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    
    if savefig and out_path:
        plt.savefig(os.path.join(out_path, f"{metric_name.lower().replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    if show_fig:
        plt.show()
