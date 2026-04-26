
# Assignment 3 — SIS on Complex Networks

This folder contains the code used to simulate the **SIS (Susceptible–Infected–Susceptible)** epidemic model on synthetic networks (Erdös–Rényi and Barabási–Albert), compare multiple implementations (**Python baseline**, **Python optimized**, **C++ optimized**), and optionally overlay **theoretical predictions** (MMCA and QMF).

The typical workflow is:

1. Export the networks to `networks/`.
2. Run simulations (Python normal / Python efficient / C++).
3. Generate theoretical curves into `results/`.
4. Plot selected results into `assets/`.

## Project layout (important files)

- `main.py`
	- Runs SIS simulations on 4 graphs (ER/BA with ⟨k⟩≈4 and ⟨k⟩≈6).
	- Writes CSV files under `results/`.
	- Supports an `--efficient` flag to use the optimized Python implementation.

- `sis.py`
	- Contains the two Python SIS implementations:
		- `SIS_simulation(...)`: baseline, node-by-node infection checks.
		- `SIS_simulation_optimized(...)`: set-based optimized approach.

- `export_networks.py`
	- Exports the same 4 graphs (with fixed random seeds) to `networks/*.txt`.
	- These text files are required by the C++ implementation and by the theoretical predictions.

- `cpp/`
	- `sis_simulation.cpp`: C++ version matching `SIS_simulation_optimized`.
	- `Makefile`: builds the executable `sis_simulation`.
	- Outputs CSV files into `results/` (from the Assignment 3 root).

- `theoretical_predictions.py`
	- Computes theoretical stationary prevalence curves (MMCA and QMF) from `networks/*.txt`.
	- Writes `*_theoretical.csv` into `results/`.

- `plot_results.py`
	- Plots any subset of result CSVs from `results/` into `assets/`.
	- Can overlay theoretical points if the corresponding `*_theoretical.csv` exist.

## Parameters used throughout

Unless you edit the scripts, the default setup is:

- Graphs: `er_k4`, `er_k6`, `ba_k4`, `ba_k6`
- Nodes: `N = 1000`
- Infection rates: `beta = 0.00, 0.01, ..., 0.30`
- Recovery rates: `mu ∈ {0.2, 0.4}`
- Monte Carlo repetitions: `N_rep = 100`
- Time horizon: `T_max = 1000`, transient ignored: `T_trans = 900`
- Random seeds: `42` (for reproducibility)

## How to run

All commands below assume you are **inside this folder**:

```bash
cd "Assignments/Assignment 3"
```

### 1) Export networks (recommended)

Creates `networks/er_k4.txt`, `networks/er_k6.txt`, `networks/ba_k4.txt`, `networks/ba_k6.txt`.

```bash
python export_networks.py
```

Network file format:

- First line: `N E`
- Then `E` lines: `u v` (undirected edge list, 0-indexed)

### 2) Run SIS simulations (Python)

**Baseline Python** (slower):

```bash
python main.py
```

**Optimized Python** (faster):

```bash
python main.py --efficient
```

Outputs (examples):

- `results/er_k4_mu_0.2.csv`
- `results/er_k4_mu_0.2_efficient.csv`

Each CSV contains:

- `beta`: infection probability
- `rho`: average stationary fraction of infected nodes
- `time_seconds`: wall-clock time for that beta value

### 3) Run SIS simulations (C++)

Build:

```bash
make -C cpp
```

Run the executable **from the Assignment 3 root** (so it can find `networks/`):

```bash
./cpp/sis_simulation
```

Outputs (examples):

- `results/er_k4_mu_0.2_cpp.csv`
- `results/ba_k6_mu_0.4_cpp.csv`

### 4) Generate theoretical predictions (MMCA + QMF)

This reads `networks/*.txt` and writes `*_theoretical.csv` files to `results/`.

```bash
python theoretical_predictions.py
```

Outputs (examples):

- `results/er_k4_mu_0.2_theoretical.csv`

Each theoretical CSV contains:

- `beta`
- `rho_mmca`: MMCA stationary prevalence (mean over nodes)
- `rho_qmf`: QMF stationary prevalence (mean over nodes)
- `beta_c`: epidemic threshold estimate $\beta_c = \mu / \lambda_{\max}(A)$

### 5) Plot results

By default, `plot_results.py` searches `results/` and writes a PNG into `assets/`.

Plot everything it can find:

```bash
python plot_results.py
```

Plot a specific subset by passing **filters** (you can pass any mix of these tokens):

- Graph type: `er` or `ba`
- Mean degree: `4` or `6`
- Recovery: `0.2` or `0.4`
- Mode: `normal`, `efficient`, `cpp`

Examples:

```bash
# Compare all modes for ER with k=4 at mu=0.2
python plot_results.py er 4 0.2 normal efficient cpp

# Plot only C++ curves for all graphs at mu=0.4
python plot_results.py cpp 0.4
```

Overlay theoretical predictions (requires the `*_theoretical.csv` files):

```bash
python plot_results.py er 4 0.2 normal --theory
```

Optional LaTeX labels (requires a working LaTeX installation on your system):

```bash
python plot_results.py --latex
```

Generated plots are saved as:

- `assets/results_plot_*.png`

