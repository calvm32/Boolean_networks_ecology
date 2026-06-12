from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd

# 1. Define the parameter space — use your ecologically meaningful ranges
problem = {
    "num_vars": 8,
    "names": ["inf_alpha", "inf_beta", "delta",
              "awake_a", "awake_b", "T_inf",
              "T_TBD", "T_win"],
    "bounds": [
        [1,   5],       # inf_alpha
        [2,  10],       # inf_beta
        [0.005, 0.05],  # delta
        [-4,  -2],      # awake_a
        [0.3,  1.0],    # awake_b
        [10,   40],     # T_inf
        [3.9,  4.3],    # T_TBD
        [150, 210],     # T_win
    ],
}

# 2. Generate Saltelli samples  (N * (2*num_vars + 2) total runs)
#    N=128 → 128 * 18 = 2304 runs; N=64 → 1152 runs (fast for testing)
N = 128
param_values = saltelli.sample(problem, N, calc_second_order=False)

# 3. Run the model for each sample row
STEPS = 400
N0    = 100
Y_Pmax   = np.zeros(len(param_values))
Y_Sfinal = np.zeros(len(param_values))
Y_Mfinal = np.zeros(len(param_values))

base = {k: v for k, v in zip(
    ["inf_alpha","inf_beta","delta","awake_a","awake_b",
     "T_inf","T_TBD","T_AD","T_seasonal","T_win",
     "lambda_win","lambda_sum","immunity_period",
     "birth_resistance_max","recover_resistance_max"],
    [5, 2, 0.05, -3, 0.7, 30, 4.1, 88.5/1440,
     40, 210, 0.0, 0.01, 0, 0, 0.02]
)}

for i, row in enumerate(param_values):
    params = {**base}
    for name, val in zip(problem["names"], row):
        params[name] = val
        if name == "inf_alpha":
            params[name] = max(1.0, val)    # keep α > 1

    hist = simulate(make_initial_state(), steps=STEPS, parameters=params)
    m    = compute_metrics(hist, N0)
    Y_Pmax[i]   = m["P_max"]
    Y_Sfinal[i] = m["S_final"]
    Y_Mfinal[i] = m["M_final"]
    if i % 100 == 0:
        print(f"Sobol run {i}/{len(param_values)}")

# 4. Analyze
Si_P = sobol.analyze(problem, Y_Pmax,   calc_second_order=False, print_to_console=False)
Si_S = sobol.analyze(problem, Y_Sfinal, calc_second_order=False, print_to_console=False)
Si_M = sobol.analyze(problem, Y_Mfinal, calc_second_order=False, print_to_console=False)

# 5. Plot — grouped bar chart (S1 and ST side by side per parameter)
def plot_sobol(Si, problem, title, ax):
    names  = problem["names"]
    x      = np.arange(len(names))
    width  = 0.35
    ax.bar(x - width/2, Si["S1"], width, label="S1 (first-order)",
           color="#4393c3", yerr=Si["S1_conf"], capsize=3, error_kw={"lw":0.8})
    ax.bar(x + width/2, Si["ST"], width, label="ST (total)",
           color="#d6604d", yerr=Si["ST_conf"], capsize=3, error_kw={"lw":0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Sobol Index")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
plot_sobol(Si_P, problem, "Sensitivity: Peak Prevalence (P_max)",    axes[0])
plot_sobol(Si_S, problem, "Sensitivity: Final Persistence (S_final)", axes[1])
plot_sobol(Si_M, problem, "Sensitivity: Final Mortality (M_final)",   axes[2])
fig.suptitle("Sobol' Global Sensitivity Analysis", fontsize=13, y=1.02)
fig.tight_layout()
plt.savefig("figures/sobol_indices.pdf", bbox_inches="tight", dpi=300)
plt.show()