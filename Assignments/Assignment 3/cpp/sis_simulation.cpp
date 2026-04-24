#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <stdexcept>


struct Graph {
    int N;
    std::vector<std::vector<int>> adj;
};

Graph read_graph(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + filename);

    Graph g;
    int E;
    f >> g.N >> E;
    g.adj.resize(g.N);
    for (int i = 0; i < E; i++) {
        int u, v;
        f >> u >> v;
        g.adj[u].push_back(v);
        g.adj[v].push_back(u);
    }
    return g;
}

// ----------------------------------------------------------------------------
// SIS simulation (replicates SIS_simulation_optimized in sis.py)
// ----------------------------------------------------------------------------

double SIS_simulation_optimized(
    const Graph& g,
    double beta,
    double mu,
    std::mt19937& rng,
    int N_rep   = 100,
    double R0   = 0.2,
    int T_max   = 1000,
    int T_trans = 900
) {
    const int N = g.N;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    int num_initial = static_cast<int>(R0 * N);
    std::vector<int> node_order(N);
    std::iota(node_order.begin(), node_order.end(), 0);

    double average_rho = 0.0;

    for (int rep = 0; rep < N_rep; rep++) {
        // --- initial infection: random sample without replacement ---
        std::shuffle(node_order.begin(), node_order.end(), rng);

        std::vector<bool> infected(N, false);
        std::unordered_set<int> infected_set;
        infected_set.reserve(num_initial * 2);

        for (int i = 0; i < num_initial; i++) {
            infected[node_order[i]] = true;
            infected_set.insert(node_order[i]);
        }

        double avg_rho_per_timestep = 0.0;

        for (int t = 0; t < T_max; t++) {
            if (infected_set.empty()) break;

            // Recoveries
            std::vector<int> recoveries;
            recoveries.reserve(infected_set.size());
            for (int node : infected_set)
                if (uniform(rng) < mu)
                    recoveries.push_back(node);

            // Count infected neighbours for each susceptible candidate
            std::unordered_map<int, int> candidate_count;
            candidate_count.reserve(infected_set.size() * 6);
            for (int node : infected_set)
                for (int nb : g.adj[node])
                    if (!infected[nb])
                        candidate_count[nb]++;

            // Stochastic infection
            std::vector<int> newly_infected;
            for (auto& [node, cnt] : candidate_count) {
                double prob = 1.0 - std::pow(1.0 - beta, cnt);
                if (uniform(rng) < prob)
                    newly_infected.push_back(node);
            }

            // Apply state changes
            for (int node : recoveries) {
                infected[node] = false;
                infected_set.erase(node);
            }
            for (int node : newly_infected) {
                infected[node] = true;
                infected_set.insert(node);
            }

            if (t >= T_trans)
                avg_rho_per_timestep += static_cast<double>(infected_set.size()) / N;
        }

        average_rho += avg_rho_per_timestep / (T_max - T_trans);
    }

    return average_rho / N_rep;
}


int main() {
    const int    N_rep   = 100;
    const double R0      = 0.2;
    const int    T_max   = 1000;
    const int    T_trans = 900;

    std::mt19937 rng(42);

    // betas: 0.00, 0.01, ..., 0.30  (matches np.arange(0, 0.31, 0.01))
    std::vector<double> betas;
    for (int i = 0; i <= 30; i++)
        betas.push_back(i * 0.01);

    const std::vector<std::string> graph_names = {"er_k4", "er_k6", "ba_k4", "ba_k6"};
    const std::vector<std::pair<double, std::string>> mus = {{0.2, "0.2"}, {0.4, "0.4"}};

    std::filesystem::create_directories("results");

    for (const auto& graph_name : graph_names) {
        std::cout << "\n=== " << graph_name << " ===\n";
        Graph g = read_graph("networks/" + graph_name + ".txt");
        std::cout << "  N=" << g.N << "\n";

        for (auto& [mu, mu_str] : mus) {
            std::string out_path = "results/" + graph_name + "_mu_" + mu_str + "_cpp.csv";
            std::ofstream csv(out_path);
            csv << "beta,rho,time_seconds\n";

            std::cout << "  mu=" << mu_str << "\n";

            for (double beta : betas) {
                auto t0 = std::chrono::high_resolution_clock::now();
                double rho = SIS_simulation_optimized(g, beta, mu, rng, N_rep, R0, T_max, T_trans);
                auto t1 = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(t1 - t0).count();

                csv << std::fixed << std::setprecision(6)
                    << beta << "," << rho << "," << elapsed << "\n";

                std::cout << "    beta=" << std::fixed << std::setprecision(2) << beta
                          << "  rho=" << std::fixed << std::setprecision(4) << rho
                          << "  t=" << std::fixed << std::setprecision(2) << elapsed << "s\n";
            }

            std::cout << "  -> " << out_path << "\n";
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
