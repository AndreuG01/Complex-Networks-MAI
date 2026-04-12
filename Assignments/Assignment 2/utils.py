import os
from matplotlib.colors import ListedColormap
import networkx as nx
import matplotlib.pyplot as plt

def extract_filename_info(filename: str) -> dict:
    splitted_elems = filename.split("_")[2:]
    res = {}
    i = 0
    while i < len(splitted_elems):
        key = splitted_elems[i]
        value = splitted_elems[i + 1]
        res[key] = float(value)
        i += 2

    return res

def generate_colormap():
    # Custom color palette generated with AI tools, and refined to ensure good contrast and visibility for community visualization.
    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
        "#00FFFF", "#FFA500", "#800080", "#008000", "#800000",
        "#000080", "#808000", "#008080", "#FFC0CB", "#A52A2A",
        "#FF4500", "#2E8B57", "#4682B4", "#9ACD32", "#DAA520",
        "#DC143C", "#00CED1", "#7B68EE", "#FF1493", "#32CD32",
        "#FF8C00", "#8B008B", "#48D1CC", "#FF6347", "#40E0D0",
        "#EE82EE", "#F4A460", "#5F9EA0", "#D2691E", "#B8860B"
    ]
    return ListedColormap(colors)
    
    
def match_communities(reference_comms, target_comms):
    reference_sets = [set(c) for c in reference_comms]
    target_sets = [set(c) for c in target_comms]

    mapping = {}
    used_reference = set()

    for i, target in enumerate(target_sets):
        best_j = None
        best_score = -1.0

        for j, reference in enumerate(reference_sets):
            if j in used_reference:
                continue

            union = len(target | reference)
            if union == 0:
                continue

            score = len(target & reference) / union

            if score > best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            mapping[i] = best_j
            used_reference.add(best_j)
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


def save_partition_to_clu(G, communities, filepath):
    nodes_sorted = sorted(G.nodes(), key=lambda x: int(x))
    node_to_comm = {}

    for comm_idx, comm in enumerate(communities, start=1):
        for node in comm:
            node_to_comm[str(node)] = comm_idx

    with open(filepath, "w") as f:
        f.write(f"*Vertices {len(nodes_sorted)}\n")
        for node in nodes_sorted:
            f.write(f"{node_to_comm[str(node)]}\n")


def load_partition_from_clu(G, filepath):
    nodes_sorted = sorted(G.nodes(), key=lambda x: int(x))

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Skip header (*Vertices N)
    labels = [int(line.strip()) for line in lines[1:]]

    if len(labels) != len(nodes_sorted):
        raise ValueError("Mismatch between number of nodes and labels in .clu file")

    communities_dict = {}
    for node, comm_id in zip(nodes_sorted, labels):
        if comm_id not in communities_dict:
            communities_dict[comm_id] = []
        communities_dict[comm_id].append(node)

    communities = list(communities_dict.values())
    return communities