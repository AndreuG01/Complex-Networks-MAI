import networkx as nx
from networkx.algorithms.community import louvain_communities
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import numpy as np
from utils import generate_colormap, match_communities, reorder_communities

plt.rcParams.update({
    "text.usetex": True,
})


DATA_PATH = "data"
PRIMARY_SCHOOL_NETWORKS = "primary_school"
PRIMARY_SCHOOL_WEIGHTED = "primaryschool_w.net"
METADATA = "metadata_primary_school.txt"
ASSETS = "assets"
os.makedirs(ASSETS, exist_ok=True)

def load_metadata():
    metadata_path = os.path.join(DATA_PATH, PRIMARY_SCHOOL_NETWORKS, METADATA)
    metadata = {}
    with open(metadata_path, 'r') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(" ")
            if len(parts) >= 2:
                key, value = parts[0], parts[1]
                metadata[key.strip()] = value.strip()
    return metadata


def metadata_to_communities(G, metadata):
    communities = {}
    node_to_group = {}

    for node in G.nodes():
        group = metadata.get(str(node), "Unknown")
        communities.setdefault(group, []).append(node)
        node_to_group[node] = group

    return node_to_group, communities


PRIMARY_SCHOOL_UNWEIGHTED = "primaryschool_u.net"


def detect_communities(G, use_weights=True):
    if use_weights:
        communities = louvain_communities(G, weight="weight")
    else:
        communities = louvain_communities(G, weight=None)

    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i

    return node_to_comm, communities



def draw_communities(G, pos, node_to_group, title, output_path, show_fig=False, group_to_color=None):
    groups = sorted(set(node_to_group.values()))
    cmap = generate_colormap()

    if group_to_color is None:
        group_to_color = {group: cmap(i % cmap.N) for i, group in enumerate(groups)}
    else:
        group_to_color = dict(group_to_color)
        next_color_idx = len(group_to_color)
        for group in groups:
            if group not in group_to_color:
                group_to_color[group] = cmap(next_color_idx % cmap.N)
                next_color_idx += 1

    node_colors = [group_to_color[node_to_group[n]] for n in G.nodes()]

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label=group, markerfacecolor=group_to_color[group], markersize=8)
        for group in groups
    ]

    plt.figure(figsize=(7, 7))
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=80, edgecolors="black")

    plt.title(title)
    plt.axis("off")
    plt.legend(handles=legend_handles, title="School group")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{title.replace(' ', '_').replace('(', '').replace(')', '').lower()}.png"), dpi=300, bbox_inches="tight")
    if show_fig:
        plt.show()



def community_composition(communities, metadata):
    composition = []

    for comm in communities:
        counter = defaultdict(int)
        for node in comm:
            group = metadata[str(node)]
            counter[group] += 1
        composition.append(counter)

    return composition


def plot_composition(composition, title, output_path, show_fig=False):
    all_groups = sorted({g for comp in composition for g in comp.keys()})

    data = []
    for comp in composition:
        row = [comp.get(g, 0) for g in all_groups]
        data.append(row)

    data = np.array(data)

    bottoms = np.zeros(len(composition))

    plt.figure(figsize=(10, 5))

    for i, group in enumerate(all_groups):
        plt.bar(range(len(composition)), data[:, i], bottom=bottoms, label=group, color=generate_colormap()(i))
        bottoms += data[:, i]

    plt.xlabel("Communities")
    plt.ylabel("Number of nodes")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{title.replace(' ', '_').replace('(', '').replace(')', '').lower()}.png"), dpi=300, bbox_inches="tight")
    if show_fig:
        plt.show()


if __name__ == "__main__":
    weighted_path = os.path.join(DATA_PATH, PRIMARY_SCHOOL_NETWORKS, PRIMARY_SCHOOL_WEIGHTED)
    unweighted_path = os.path.join(DATA_PATH, PRIMARY_SCHOOL_NETWORKS, PRIMARY_SCHOOL_UNWEIGHTED)

    Gw = nx.read_pajek(weighted_path)
    Gu = nx.read_pajek(unweighted_path)

    metadata = load_metadata()

    # Ensure simple graphs
    if Gw.is_multigraph():
        Gw = nx.Graph(Gw)
    if Gu.is_multigraph():
        Gu = nx.Graph(Gu)

    
    for u, v, d in Gw.edges(data=True):
        w = d["weight"]
        d["inv_weight"] = 1.0 / w if w > 0 else 1.0

    pos = nx.kamada_kawai_layout(Gw, weight="inv_weight")

    
    node_to_comm_w, comms_w = detect_communities(Gw, use_weights=True)
    node_to_comm_u, comms_u = detect_communities(Gu, use_weights=False)

    aligned_u_to_w = match_communities(comms_w, comms_u)
    node_to_comm_u_aligned = {
        node: aligned_u_to_w.get(comm, comm)
        for node, comm in node_to_comm_u.items()
    }

    shared_cmap = generate_colormap()
    shared_group_to_color = {
        i: shared_cmap(i % shared_cmap.N)
        for i in range(max(len(comms_w), len(comms_u)))
    }

    draw_communities(Gw, pos, node_to_comm_w, "Weighted Network Communities", ASSETS, group_to_color=shared_group_to_color)
    draw_communities(Gu, pos, node_to_comm_u_aligned, "Unweighted Network Communities", ASSETS, group_to_color=shared_group_to_color)

    node_to_group, original_communities = metadata_to_communities(Gw, metadata)
    draw_communities(Gw, pos, node_to_group, "Primary School Network (Original Communities)", ASSETS)

    comp_w = community_composition(comms_w, metadata)
    comp_u = community_composition(comms_u, metadata)

    plot_composition(comp_w, "Community Composition (Weighted)", ASSETS)
    plot_composition(comp_u, "Community Composition (Unweighted)", ASSETS)