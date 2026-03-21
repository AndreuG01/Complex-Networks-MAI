import os
import matplotlib.pyplot as plt
import networkx as nx

DATA_PATH = "data"
OUT_PATH = "assets"
os.makedirs(OUT_PATH, exist_ok=True)


USE_LATEX = True

if USE_LATEX:
    plt.rcParams.update({
        "text.usetex": True,
    })
    

colors = [
    "#640D5F",
    "#D91656",
    "#EB5B00",
    "#FFB200",
    "#2D728F",
    "#3E8914",
    "#FF00D3",
]



def load_network(net_name: str, data_path: str=DATA_PATH) -> nx.Graph:
    net_data = nx.read_pajek(os.path.join(data_path, net_name))
    G = nx.Graph(net_data)
    return G
