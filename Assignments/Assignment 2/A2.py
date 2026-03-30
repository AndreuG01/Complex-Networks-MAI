import pandas as pd
import numpy as np
from tqdm.asyncio import tqdm
from utils import extract_filename_info, generate_colormap
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from collections import Counter
import glob
import os
import re
import infomap as im

plt.rcParams.update({
    "text.usetex": True
})

def match_communities(reference_comms, target_comms):
    ref_sets = [set(c) for c in reference_comms]
    tgt_sets = [set(c) for c in target_comms]

    mapping = {}
    used_ref = set()

    for i, tgt in enumerate(tgt_sets):
        best_j = None
        best_score = -1.0

        for j, ref in enumerate(ref_sets):
            if j in used_ref:
                continue

            union = len(tgt | ref)
            if union == 0:
                continue

            score = len(tgt & ref) / union

            if score > best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            mapping[i] = best_j
            used_ref.add(best_j)
        else:
            mapping[i] = i

    return mapping


def reorder_communities(reference_comms, target_comms):
    mapping = match_communities(reference_comms, target_comms)

    reordered = [None] * max(len(reference_comms), len(target_comms))

    for i, comm in enumerate(target_comms):
        new_idx = mapping.get(i, i)
        if new_idx >= len(reordered):
            reordered.extend([None] * (new_idx - len(reordered) + 1))
        reordered[new_idx] = comm

    return [c for c in reordered if c is not None]



def plot_degree_distributions(
    G, filepath=None, prr=None, out_path=None, savefig=True, show_fig=False
):
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



def detect_communities(G, pos, filepath=None, prr=None, algorithm_name='louvain'):
    G_simple = G
    
    communities = []
    if algorithm_name.lower() == 'greedy':
        communities = [list(com) for com in nx.community.greedy_modularity_communities(G_simple)]
        
    elif algorithm_name.lower() == 'louvain':
        modularity_louvain = -1
        best_communities_louvain = None
        for realization in range(10):
            comms = nx.community.louvain_communities(G_simple, seed=realization)
            mod = nx.community.modularity(G_simple, comms)
            if mod > modularity_louvain:
                modularity_louvain = mod
                best_communities_louvain = comms
        communities = [list(com) for com in best_communities_louvain]
        
    elif algorithm_name.lower() == 'infomap':
        # Infomap expects integer node IDs, so we map string IDs to integers and back
        mapping = {node: i for i, node in enumerate(G_simple.nodes())}
        reverse_mapping = {i: node for node, i in mapping.items()}
        
        im_model = im.Infomap(silent=True)
        for u, v in G_simple.edges():
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
    missing_nodes = [node for node in G_simple.nodes() if node not in set(assigned_nodes)]
    if missing_nodes:
        communities.extend([[node] for node in missing_nodes])
        assigned_nodes = [node for comm in communities for node in comm]


    return communities

    # CALCULATE MODULARITY
    num_communities = len(communities)
    modularity = nx.community.modularity(G_simple, communities)
    
    

def plot_communities_evolution():
    pass

DATA_PATH = "data"
SYNTHETIC_NETWORKS = "synthetic"
PRIMARY_SCHOOL_NETWORKS = "primary_school"
ASSETS = "assets"
os.makedirs(ASSETS, exist_ok=True)
INITIAL_NETWORK_VISUALIZATIONS = os.path.join(ASSETS, "initial_network_visualizations")
os.makedirs(INITIAL_NETWORK_VISUALIZATIONS, exist_ok=True)

DEGREE_DISTRIBUTION_PLOTS = os.path.join(ASSETS, "degree_distribution_plots")
os.makedirs(DEGREE_DISTRIBUTION_PLOTS, exist_ok=True)

GREEDY_COMMUNITY_VISUALIZATIONS = os.path.join(ASSETS, "greedy_community_visualizations")
os.makedirs(GREEDY_COMMUNITY_VISUALIZATIONS, exist_ok=True)

LOUVAIN_COMMUNITY_VISUALIZATIONS = os.path.join(ASSETS, "louvain_community_visualizations")
os.makedirs(LOUVAIN_COMMUNITY_VISUALIZATIONS, exist_ok=True)

INFOMAP_COMMUNITY_VISUALIZATIONS = os.path.join(ASSETS, "infomap_community_visualizations")
os.makedirs(INFOMAP_COMMUNITY_VISUALIZATIONS, exist_ok=True)

if __name__ == "__main__":
    synth_net_path = os.path.join(DATA_PATH, SYNTHETIC_NETWORKS)

    reference_pos = None

    file = sorted(os.listdir(synth_net_path))[-1]
    filename = os.path.splitext(file)[0] # Remove the filename extension
    print(f"Calculating reference layout using: {file}")
    G_ref = nx.read_pajek(os.path.join(synth_net_path, file))
    reference_pos = nx.spring_layout(G_ref, seed=42) 

    files = sorted(os.listdir(synth_net_path))
    max_communities = 0
    for file in tqdm(files, desc="Processing synthetic networks", total=len(files)):
        filename = os.path.splitext(file)[0]
        prr_value = extract_filename_info(filename).get('prr', None)    
        G = nx.read_pajek(os.path.join(synth_net_path, file))

        # visualize_network(
        #     G, pos=reference_pos,
        #     out_path=INITIAL_NETWORK_VISUALIZATIONS,
        #     savefig=True,
        #     show_fig=False,
        #     prr=prr_value
        # )
        
        # plot_degree_distributions(
        #     G,
        #     filepath=file,
        #     out_path=DEGREE_DISTRIBUTION_PLOTS,
        #     savefig=True,
        #     show_fig=False,
        #     prr=prr_value,
        # )

        communities_greedy = detect_communities(G, pos=reference_pos, filepath=file, prr=prr_value, algorithm_name='greedy')
        communities_louvain = detect_communities(G, pos=reference_pos, filepath=file, prr=prr_value, algorithm_name='louvain')
        communities_infomap = detect_communities(G, pos=reference_pos, filepath=file, prr=prr_value, algorithm_name='infomap')
        
        # Align community indices across algorithms for better visual comparison
        communities_louvain = reorder_communities(communities_greedy, communities_louvain)
        communities_infomap = reorder_communities(communities_greedy, communities_infomap)
        
        # visualize_network(
        #     G, pos=reference_pos,
        #     out_path=GREEDY_COMMUNITY_VISUALIZATIONS,
        #     savefig=True,
        #     show_fig=False,
        #     prr=prr_value,
        #     communities=communities_greedy
        # )
        # visualize_network(
        #     G, pos=reference_pos,
        #     out_path=LOUVAIN_COMMUNITY_VISUALIZATIONS,
        #     savefig=True,
        #     show_fig=False,
        #     prr=prr_value,
        #     communities=communities_louvain
        # )
        
        # visualize_network(
        #     G, pos=reference_pos,
        #     out_path=INFOMAP_COMMUNITY_VISUALIZATIONS,
        #     savefig=True,
        #     show_fig=False,
        #     prr=prr_value,
        #     communities=communities_infomap
        # )
        
        
            
        