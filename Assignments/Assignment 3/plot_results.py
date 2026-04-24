import argparse
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import os


RESULTS_DIR = "results"
OUTPUT_DIR = "assets"

VALID_GRAPHS = {"ba", "er"}
VALID_KS = {"4", "6"}
VALID_MUS = {"0.2", "0.4"}
VALID_MODES = {"normal", "efficient", "cpp"}

MODE_SUFFIX = {
    "normal": "",
    "efficient": "_efficient",
    "cpp": "_cpp",
}


def parse_filters(raw_filters):
    selected_graphs = []
    selected_ks = []
    selected_mus = []
    selected_modes = []

    for token in raw_filters:
        value = token.strip().lower()
        if value in VALID_GRAPHS:
            selected_graphs.append(value)
        elif value in VALID_KS:
            selected_ks.append(value)
        elif value in VALID_MUS:
            selected_mus.append(value)
        elif value in VALID_MODES:
            selected_modes.append(value)
        else:
            raise ValueError(f"Unsupported filter '{token}'. Allowed values are ba/er, 4/6, 0.2/0.4, and normal/efficient/cpp.")

    return (
        selected_graphs or sorted(VALID_GRAPHS),
        selected_ks or sorted(VALID_KS),
        selected_mus or sorted(VALID_MUS),
        selected_modes or sorted(VALID_MODES),
    )


def build_filename(graph, k, mu, mode):
    suffix = MODE_SUFFIX[mode]
    return f"{graph}_k{k}_mu_{mu}{suffix}.csv"


def load_result_file(path):
    frame = pd.read_csv(path)
    required_columns = {"beta", "rho"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing_columns)}")
    return frame.sort_values("beta")


def load_theory_file(graph, k, mu):
    path = os.path.join(RESULTS_DIR, f"{graph}_k{k}_mu_{mu}_theoretical.csv")
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    return frame.sort_values("beta")


def make_label(graph, k, mu, mode, use_latex=False):
    if use_latex:
        label = f"{graph.upper()} $k={k}$, $\\mu={mu}$"
    else:
        label = f"{graph.upper()} k={k}, mu={mu}"

    if mode != "normal":
        label += f", {mode}"
    return label


def build_output_name(graphs, ks, mus, modes, show_theory=False):
    def join_values(values, prefix):
        if len(values) == 1:
            return f"{prefix}{values[0]}"
        return f"{prefix}all"

    def join_modes(values):
        if len(values) == 1:
            return f"mode-{values[0]}"
        if len(values) == len(VALID_MODES):
            return "mode-all"
        return "mode-" + "-".join(values)

    parts = [
        join_values(graphs, "graph-"),
        join_values(ks, "k-"),
        join_values(mus, "mu-"),
        join_modes(modes),
    ]
    name = "results_plot_" + "_".join(parts)
    if show_theory:
        name += "_theory"
    return name + ".png"


def plot_results(raw_filters, use_latex=False, show_theory=False):
    graphs, ks, mus, modes = parse_filters(raw_filters)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({"text.usetex": use_latex})

    combinations = list(product(graphs, ks, mus, modes))
    matched_files = []

    for graph, k, mu, mode in combinations:
        filename = build_filename(graph, k, mu, mode)
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            matched_files.append((graph, k, mu, mode, path))

    if not matched_files:
        raise FileNotFoundError("No matching result files were found for the requested filters.")
    
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)


    colors = plt.cm.tab20.colors
    theory_plotted = set()

    for index, (graph, k, mu, mode, path) in enumerate(matched_files):
        color = colors[index % len(colors)]
        data = load_result_file(path)
        ax.plot(
            data["beta"],
            data["rho"],
            label=make_label(graph, k, mu, mode, use_latex=use_latex),
            linewidth=2.2,
            color=color,
            marker="o",
            markersize=3.2,
            markevery=max(len(data) // 8, 1),
            alpha=0.95,
        )

        if show_theory and (graph, k, mu) not in theory_plotted:
            theory = load_theory_file(graph, k, mu)
            if theory is not None:
                base = f"{graph.upper()} k={k}, mu={mu}" if not use_latex else f"{graph.upper()} $k={k}$, $\\mu={mu}$"
                ax.scatter(
                    theory["beta"], theory["rho_mmca"],
                    label=f"{base} MMCA",
                    color=color, marker="o", s=18, zorder=5,
                )
                ax.scatter(
                    theory["beta"], theory["rho_qmf"],
                    label=f"{base} QMF",
                    color=color, marker="s", s=18, zorder=5, alpha=0.6,
                )
                theory_plotted.add((graph, k, mu))

    title = r"SIS Simulation Results" if use_latex else "SIS Simulation Results"
    x_label = r"$\beta$" if use_latex else "beta"
    y_label = r"$\rho$" if use_latex else "rho"

    ax.set_title(title, pad=14, fontsize=15)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, ncol=2, frameon=True)
    # Use a grid
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, build_output_name(graphs, ks, mus, modes, show_theory=show_theory))
    fig.savefig(output_path, bbox_inches="tight")
    # plt.show()

    print(f"Loaded {len(matched_files)} file(s).")
    print(f"Saved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot SIS result CSV files from the results folder. "
            "Pass any mix of ba/er, 4/6, 0.2/0.4, and normal/efficient/cpp. "
            "Use --latex to render text with LaTeX."
        )
    )
    parser.add_argument(
        "filters",
        nargs="*",
        help="Optional filters: ba/er, 4/6, 0.2/0.4, normal/efficient/cpp",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Enable LaTeX text rendering in plot labels and title (default: disabled).",
    )
    parser.add_argument(
        "--theory",
        action="store_true",
        help="Overlay MMCA and QMF theoretical predictions as dots (requires _theoretical.csv files).",
    )
    args = parser.parse_args()
    plot_results(args.filters, use_latex=args.latex, show_theory=args.theory)


if __name__ == "__main__":
    main()