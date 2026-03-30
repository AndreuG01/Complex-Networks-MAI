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
    
    


if __name__ == "__main__":
    pass