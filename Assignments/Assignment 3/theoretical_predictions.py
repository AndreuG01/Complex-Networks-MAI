#TODO: review
"""
Theoretical predictions for SIS epidemic spreading on quenched networks.

MMCA (Microscopic Markov Chain Approach)
-----------------------------------------
Per-node update at each iteration:
    rho_i <- rho_i * (1 - mu) + (1 - rho_i) * [1 - prod_{j in N(i)} (1 - beta * rho_j)]

The product is evaluated efficiently via the log-sum trick:
    prod_{j} (1 - beta*rho_j) = exp( A @ log(1 - beta*rho) )

QMF (Quenched Mean Field)
--------------------------
Steady-state condition:  mu * rho_i = beta * (1 - rho_i) * (A @ rho)_i
Solved by the self-consistent iteration:
    rho_i <- beta * theta_i / (mu + beta * theta_i),  theta = A @ rho

Epidemic threshold (both approaches linearize to the same condition):
    beta_c = mu / lambda_max(A)

Results are saved to results/<graph>_mu_<mu>_theoretical.csv with columns:
    beta, rho_mmca, rho_qmf, beta_c
"""

import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


NETWORK_DIR = "networks"
RESULTS_DIR = "results"


def read_graph(path: str) -> nx.Graph:
    with open(path) as f:
        N, E = map(int, f.readline().split())
        edges = [tuple(map(int, f.readline().split())) for _ in range(E)]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    return G


def build_sparse_adj(G: nx.Graph) -> sp.csr_matrix:
    N = G.number_of_nodes()
    rows, cols = zip(*((u, v) for u, v in G.edges())) if G.number_of_edges() else ([], [])
    data = np.ones(len(rows), dtype=float)
    A = sp.csr_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(N, N),
    )
    return A


def spectral_radius(A: sp.csr_matrix) -> float:
    vals = spla.eigsh(A, k=1, which="LM", return_eigenvectors=False)
    return float(vals[0])


def mmca_rho(A: sp.csr_matrix, beta: float, mu: float, tol: float = 1e-10, max_iter: int = 20000) -> float:
    N = A.shape[0]
    rho = np.full(N, 0.5)
    for _ in range(max_iter):
        log_no_inf = np.log(np.maximum(1.0 - beta * rho, 1e-300))
        q_inf = 1.0 - np.exp(A @ log_no_inf)
        rho_new = np.clip(rho * (1.0 - mu) + (1.0 - rho) * q_inf, 0.0, 1.0)
        if np.max(np.abs(rho_new - rho)) < tol:
            return float(np.mean(rho_new))
        rho = rho_new
    return float(np.mean(rho))


def qmf_rho(A: sp.csr_matrix, beta: float, mu: float, tol: float = 1e-10, max_iter: int = 20000) -> float:
    N = A.shape[0]
    rho = np.full(N, 0.5)
    for _ in range(max_iter):
        theta = A @ rho
        denom = mu + beta * theta
        rho_new = np.clip(beta * theta / np.where(denom > 0, denom, 1e-300), 0.0, 1.0)
        if np.max(np.abs(rho_new - rho)) < tol:
            return float(np.mean(rho_new))
        rho = rho_new
    return float(np.mean(rho))


def main() -> None:
    betas = np.arange(0.0, 0.31, 0.01)
    mus = [0.2, 0.4]
    graph_names = ["er_k4", "er_k6", "ba_k4", "ba_k6"]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for name in graph_names:
        path = os.path.join(NETWORK_DIR, f"{name}.txt")
        print(f"\n=== {name} ===")
        G = read_graph(path)
        N, E = G.number_of_nodes(), G.number_of_edges()
        print(f"  N={N}, E={E}, <k>={2*E/N:.2f}")

        A = build_sparse_adj(G)
        lmax = spectral_radius(A)
        print(f"  lambda_max = {lmax:.4f}")

        for mu in mus:
            beta_c = mu / lmax
            print(f"  mu={mu}  beta_c={beta_c:.4f}")

            rows = []
            for beta in betas:
                rho_m = mmca_rho(A, beta, mu)
                rho_q = qmf_rho(A, beta, mu)
                rows.append({
                    "beta": round(float(beta), 6),
                    "rho_mmca": rho_m,
                    "rho_qmf": rho_q,
                    "beta_c": beta_c,
                })
                print(f"    beta={beta:.2f}  rho_mmca={rho_m:.4f}  rho_qmf={rho_q:.4f}")

            out = os.path.join(RESULTS_DIR, f"{name}_mu_{mu}_theoretical.csv")
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
