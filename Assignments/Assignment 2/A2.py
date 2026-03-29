import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
import networkx as nx
from collections import Counter
import glob
import os
import re
import infomap as im

rcParams['figure.figsize'] = (5, 5)
rcParams['font.size']=16



def plot_degree_distribution(G,scale='linear',rep='bars',weight=False, filepath = None, prr=None):
    if(weight==False):
        degree_sequence=[G.degree(node) for node in G.nodes()]
    else:
        degree_sequence=[G.degree(node,weight='weight') for node in G.nodes()]
        
    degree_counts = Counter(degree_sequence)
    min_degree=min(degree_sequence)
    max_degree=max(degree_sequence)

    degrees=list(degree_counts.keys())
    degree_count=list(degree_counts.values())

    fig,ax=plt.subplots(1,1,figsize=(5,5))
    if rep=='bars':
        ax.bar(degrees,degree_count)
    if rep=='scatter':
        ax.scatter(degrees,degree_count)
    
    if scale=='log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    if(weight==False):
        ax.set_xlabel('Degree',fontsize=15)
    else:
        ax.set_xlabel('Strength',fontsize=15)
    ax.set_ylabel('#nodes',fontsize=15)
    ax.tick_params(which='major',axis='both',labelsize=15)  

    if prr_value is not None:
        ax.set_title(f'prr = {prr}', fontsize=16)  

    if filepath:
        save_dir = 'deg_distr'
        os.makedirs(save_dir, exist_ok=True)
        base_filename = os.path.basename(filepath)
        new_filename = base_filename.replace('.net', '_degree_distribution.png')
        save_path = os.path.join(save_dir, new_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def visualize_network(G, pos, filepath=None, prr=None):
    fig, ax = plt.subplots(figsize=(5, 5))

    # sizes = [10 + 5 * G.degree(node, weight='weight') for node in G.nodes()]

    nx.draw(G, pos, ax=ax, with_labels=False, node_color='blue', edge_color='gray', node_size=20)
    
    if prr is not None:
        ax.set_title(f'prr = {prr}', fontsize=16)  

    if filepath:
        save_dir = 'initial_network_visualizations'
        os.makedirs(save_dir, exist_ok=True)
        base_filename = os.path.basename(filepath)
        new_filename = base_filename.replace('.net', '_visualization.png')
        save_path = os.path.join(save_dir, new_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

import infomap as im # Make sure this is installed: pip install infomap

def detect_communities(G, pos, filepath=None, prr=None, algorithm_name='louvain'):

    # Community detection algorithms in NetworkX require standard simple Graphs
    G_simple = nx.Graph(G)
    
    # RUN COMMUNITY DETECTION ALGORITHM
    if algorithm_name.lower() == 'greedy':
        communities = list(nx.community.greedy_modularity_communities(G_simple))
        
    elif algorithm_name.lower() == 'louvain':
        modularity_louvain = -1
        best_communities_louvain = None
        for realization in range(10):
            comms = nx.community.louvain_communities(G_simple, seed=realization)
            mod = nx.community.modularity(G_simple, comms)
            if mod > modularity_louvain:
                modularity_louvain = mod
                best_communities_louvain = comms
        communities = best_communities_louvain
        
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
        communities = list(comm_dict.values())
        
    else:
        print(f"Algorithm '{algorithm_name}' not recognized.")
        return

    # CALCULATE MODULARITY
    num_communities = len(communities)
    modularity = nx.community.modularity(G_simple, communities)
    
    # SAVE TO TEXT FILE
    metrics_file = f'community_metrics_{algorithm_name.lower()}.txt'
    
    file_exists = os.path.isfile(metrics_file)
    
    with open(metrics_file, 'a') as f:
                    
        f.write(f"{prr}\t{num_communities}\t{modularity:.4f}\n")
        
    # 4. VISUALIZE AND SAVE PLOT
    # Create a dictionary mapping each node to its community index for coloring
    community_index = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_index[node] = i
            
    # List of colors for nodes in exactly the same order as G.nodes()
    color_nodes = [community_index[node] for node in G.nodes()]
    
    fig, ax = plt.subplots(figsize=(5, 5))
    # sizes = [15 + 3 * G_simple.degree(node) for node in G_simple.nodes()]
    
    # cmap=plt.cm.tab20 gives us up to 20 highly distinct colors for communities
    nx.draw(G_simple, pos, ax=ax, with_labels=False, node_color=color_nodes, 
            cmap=plt.cm.tab20, edge_color='lightgrey', node_size=20)
            
    if prr is not None:
        ax.set_title(f'{algorithm_name.capitalize()} (prr = {prr})', fontsize=16)
        
    ax.margins(0.10)
    
    if filepath:
        save_dir = f'community_visualizations_{algorithm_name.lower()}'
        os.makedirs(save_dir, exist_ok=True)
        base_filename = os.path.basename(filepath)
        new_filename = base_filename.replace('.net', f'_{algorithm_name}_communities.png')
        save_path = os.path.join(save_dir, new_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def plot_communities_evolution():
    pass


data_path = directory_path = r'data/A3_synthetic_networks'

file_pattern = os.path.join(directory_path, '*.net')
network_files = glob.glob(file_pattern)

reference_pos = None

for file_path in network_files:
    match = re.search(r'prr_([0-9.]+)', file_path)
    if match and float(match.group(1)) == 1.0:
        print(f"Calculating reference layout using: {os.path.basename(file_path)}")
        G_ref = nx.read_pajek(file_path)
        # We calculate the layout once here. seed=42 ensures it's perfectly reproducible.
        reference_pos = nx.spring_layout(G_ref, seed=42) 
        break

# Fallback just in case prr=1.00 is missing from your folder
if reference_pos is None:
    print("Warning: prr=1.00 network not found! Using the last network as layout reference.")
    G_ref = nx.read_pajek(network_files[-1])
    reference_pos = nx.spring_layout(G_ref, seed=42)


# 2. RUN MAIN LOOP USING REFERENCE POSITIONS
for file_path in network_files:
    match = re.search(r'prr_([0-9.]+)', file_path)
    
    if match:
        prr_value = match.group(1)
        
        G = nx.read_pajek(file_path)
        
        print(f"***** prr = {prr_value} *****")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}\n")

        # plot_degree_distribution(G, scale='linear', rep='bars', weight=False, filepath=file_path, prr=prr_value)

        
        # visualize_network(G, pos=reference_pos, filepath=file_path, prr=prr_value)

        detect_communities(G, pos=reference_pos, filepath=file_path, prr=prr_value, algorithm_name='greedy')
        # s'ha d'arreglar que els colors es mantinguin quan anem canviant de prr


        # detect_communities(G, pos=reference_pos, filepath=file_path, prr=prr_value, algorithm_name='louvain')
        # detect_communities(G, pos=reference_pos, filepath=file_path, prr=prr_value, algorithm_name='infomap')
        # break
    
        
        

