import networkx as nx
from networkx.algorithms.community import louvain_communities
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import numpy as np
from utils import generate_colormap, match_communities, reorder_communities


def load_metadata(data_path, primary_school_networks, metadata_file):
    metadata_path = os.path.join(data_path, primary_school_networks, metadata_file)
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





def detect_communities_primary(G, use_weights=True):
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
        for i, group in enumerate(groups)
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


def align_and_pad_communities(reference_comms, target_comms, mapping=None, n_communities=None):
    if mapping is None:
        mapping = match_communities(reference_comms, target_comms)

    if n_communities is None:
        max_mapped = max(mapping.values()) if mapping else -1
        n_communities = max(len(reference_comms), max_mapped + 1, len(target_comms))

    aligned = [set() for _ in range(n_communities)]

    for src_idx, comm in enumerate(target_comms):
        dst_idx = mapping.get(src_idx, src_idx)
        if dst_idx >= len(aligned):
            aligned.extend([set() for _ in range(dst_idx - len(aligned) + 1)])
        aligned[dst_idx].update(comm)

    return aligned


def best_match_mapping(reference_comms, target_comms):
    reference_sets = [set(c) for c in reference_comms]
    target_sets = [set(c) for c in target_comms]

    mapping = {}

    for i, target in enumerate(target_sets):
        best_j = 0
        best_score = -1.0

        for j, reference in enumerate(reference_sets):
            union = len(target | reference)
            if union == 0:
                score = 0.0
            else:
                score = len(target & reference) / union

            if score > best_score:
                best_score = score
                best_j = j

        mapping[i] = best_j

    return mapping


def ground_truth_communities_and_colors(G, metadata):
    node_to_group, group_to_nodes = metadata_to_communities(G, metadata)

    groups_sorted = sorted(group_to_nodes.keys())
    gt_comms = [set(group_to_nodes[g]) for g in groups_sorted]

    cmap = generate_colormap()
    gt_group_to_color = {g: cmap(i % cmap.N) for i, g in enumerate(groups_sorted)}
    gt_idx_to_color = {i: cmap(i % cmap.N) for i in range(len(groups_sorted))}

    return node_to_group, gt_comms, gt_group_to_color, gt_idx_to_color
