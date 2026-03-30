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